"""Convert a large volume on disk to a multiscale OME-Zarr store, without loading it into Slicer.

The conversion runs in a separate Python process (``OMEZarrConvertLib/to_omezarr.py``, which also
works on its own from the command line), reading the input in slabs: the volume can be larger
than the memory, and Slicer stays responsive while it runs. Inputs: an NRRD file (.nrrd/.nhdr),
a folder of 2D slices (TIFF, PNG, BMP, JPEG; NRecon folders with their log), a multi-page TIFF.
"""

import logging
import os
import shutil
import sysconfig

import qt

import slicer
import slicer.packaging
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

REQUIREMENTS = ("zarr>=3", "tifffile", "imagecodecs")
SHARD_MAX = 2048


class OMEZarrConvert(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Convert to OME-Zarr")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Informatics")]
        self.parent.dependencies = ["OMEZarr"]
        self.parent.contributors = ["A. Murat Maga"]
        self.parent.helpText = _(
            "Convert a large NRRD file, folder of image slices or multi-page TIFF to a multiscale OME-Zarr store on "
            "this computer, without loading it into Slicer. The volume is read in slabs, so it can be larger than "
            "the memory. Open the store with the OME-Zarr module."
        )
        self.parent.acknowledgementText = ""


#
# Logic
#


class OMEZarrConvertLogic(ScriptedLoadableModuleLogic):
    @staticmethod
    def ensureRequirements():
        missing = [r for r in REQUIREMENTS if not slicer.packaging.pip_check(r)]
        if not missing:
            return
        interactive = slicer.util.mainWindow() and not slicer.app.testingEnabled()
        if interactive and not slicer.util.confirmOkCancelDisplay(
            _("The conversion needs these Python packages: {packages}. Install them now?").format(packages=", ".join(missing))
        ):
            raise RuntimeError(f"{', '.join(missing)} not installed")
        with slicer.util.tryWithErrorDisplay(_("Failed to install {packages}").format(packages=", ".join(missing)), waitCursor=True):
            slicer.util.pip_install(missing)

    @staticmethod
    def converter():
        from OMEZarrConvertLib import to_omezarr

        return to_omezarr

    @staticmethod
    def scriptPath():
        return os.path.join(os.path.dirname(__file__), "OMEZarrConvertLib", "to_omezarr.py")

    @staticmethod
    def pythonExecutable():
        """Slicer's own Python launcher, which sets up the environment the packages need."""
        name = "PythonSlicer.exe" if os.name == "nt" else "PythonSlicer"
        candidate = os.path.join(sysconfig.get_path("scripts"), name)
        return candidate if os.path.exists(candidate) else shutil.which("PythonSlicer")

    @classmethod
    def inspect(cls, path, voxelUm=None):
        """What the converter will read: a dict with the source (or None) and the error that stops it."""
        conv = cls.converter()
        info = {"source": None, "error": None, "voxelFromInput": False, "isStore": False}
        try:
            if os.path.isdir(path) and not os.path.exists(os.path.join(path, "zarr.json")):
                info["voxelFromInput"] = conv.read_nrecon_log(path) is not None
            elif path.lower().endswith((".nrrd", ".nhdr")) or os.path.exists(os.path.join(path, "zarr.json")):
                info["voxelFromInput"] = True
                info["isStore"] = os.path.isdir(path)
            info["source"] = conv.open_source(path, voxelUm if not info["voxelFromInput"] else None, threads=1)
        except conv.ConversionError as e:
            info["error"] = str(e)
        except Exception as e:  # noqa: BLE001 - shown to the user
            logging.exception("OME-Zarr convert: reading the input failed")
            info["error"] = str(e)
        return info

    @staticmethod
    def defaultOutput(path):
        base = os.path.normpath(path)
        name = os.path.basename(base)
        for suffix in (".nrrd", ".nhdr", ".tiff", ".tif", ".ome.zarr", ".zarr"):
            if name.lower().endswith(suffix):
                name = name[: -len(suffix)]
                break
        name = name + (".rebuilt.ome.zarr" if os.path.exists(os.path.join(base, "zarr.json")) else ".ome.zarr")
        return os.path.join(os.path.dirname(base), name)

    @classmethod
    def arguments(cls, inputPath, outputPath, levels=0, voxelUm=None, shard=128):
        args = [cls.scriptPath(), inputPath, outputPath, "--shard", str(shard), "--progress"]
        if levels:
            args += ["--levels", str(levels)]
        if voxelUm:
            args += ["--voxel-size", repr(float(voxelUm))]
        return args

    @staticmethod
    def availableMemory():
        try:
            from OMEZarr import OMEZarrLogic

            return OMEZarrLogic.availableMemory()
        except Exception:  # noqa: BLE001 - only used for a warning
            return None

    @classmethod
    def convertBlocking(cls, inputPath, outputPath, levels=0, voxelUm=None, shard=128, timeoutS=600):
        """Run the conversion process and wait for it (tests, scripts). Returns (exit code, output)."""
        process = qt.QProcess()
        process.setProcessChannelMode(qt.QProcess.MergedChannels)
        process.start(cls.pythonExecutable(), cls.arguments(inputPath, outputPath, levels, voxelUm, shard))
        if not process.waitForFinished(int(timeoutS * 1000)):
            process.kill()
            process.waitForFinished(5000)
            raise TimeoutError(f"conversion did not finish in {timeoutS} s")
        return process.exitCode(), bytes(process.readAll().data()).decode(errors="replace")


#
# Widget
#


class OMEZarrConvertWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = OMEZarrConvertLogic()
        self.process = None
        self.info = None
        self.outputWasStore = False
        self.lastLines = []

        form = qt.QFormLayout()
        self.layout.addLayout(form)

        self.inputEdit = qt.QLineEdit()
        self.inputEdit.setToolTip(_("An NRRD file (.nrrd, .nhdr), a folder of image slices, a multi-page TIFF, or an "
                                    "OME-Zarr store whose coarser levels are rebuilt"))
        fileButton = qt.QPushButton(_("File..."))
        folderButton = qt.QPushButton(_("Folder..."))
        inputRow = qt.QHBoxLayout()
        inputRow.addWidget(self.inputEdit, 1)
        inputRow.addWidget(fileButton)
        inputRow.addWidget(folderButton)
        form.addRow(_("Input:"), inputRow)

        self.infoLabel = qt.QLabel("")
        self.infoLabel.wordWrap = True
        self.infoLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow("", self.infoLabel)

        self.voxelSpinBox = qt.QDoubleSpinBox()
        self.voxelSpinBox.setRange(0.0, 100000.0)
        self.voxelSpinBox.setDecimals(4)
        self.voxelSpinBox.setSuffix(" µm")
        self.voxelSpinBox.setSpecialValueText(_("required"))
        self.voxelSpinBox.setToolTip(_("Isotropic voxel size of image slices and TIFF files. NRRD files and NRecon "
                                       "folders carry their own."))
        form.addRow(_("Voxel size:"), self.voxelSpinBox)

        self.outputEdit = qt.QLineEdit()
        self.outputEdit.setToolTip(_("The .ome.zarr folder to write. An existing store there is replaced."))
        outputButton = qt.QPushButton(_("..."))
        outputRow = qt.QHBoxLayout()
        outputRow.addWidget(self.outputEdit, 1)
        outputRow.addWidget(outputButton)
        form.addRow(_("Output:"), outputRow)

        self.levelsSpinBox = qt.QSpinBox()
        self.levelsSpinBox.setRange(0, 12)
        self.levelsSpinBox.setSpecialValueText(_("automatic"))
        self.levelsSpinBox.setToolTip(_("Resolution levels, full resolution included, each half the size of the one "
                                        "before. Automatic: until the coarsest is at most 512 voxels on its longest side."))
        form.addRow(_("Levels:"), self.levelsSpinBox)

        self.shardSpinBox = qt.QSpinBox()
        self.shardSpinBox.setRange(128, SHARD_MAX)
        self.shardSpinBox.setSingleStep(128)
        self.shardSpinBox.setValue(128)
        self.shardSpinBox.setSuffix(" voxels")
        self.shardSpinBox.setToolTip(_(
            "Voxels per side of a shard, a multiple of 128. At 128 every 128-voxel chunk is its own file, the simplest "
            "choice on a local disk. Larger shards group the chunks into fewer, bigger files: use them for very large "
            "volumes, network drives, synced folders or a store going to object storage. The conversion reads one "
            "shard's depth of the input at a time, so larger shards need more memory."))
        form.addRow(_("Shard size:"), self.shardSpinBox)

        self.openCheckBox = qt.QCheckBox(_("Open the store when done"))
        self.openCheckBox.checked = True
        form.addRow("", self.openCheckBox)

        self.convertButton = qt.QPushButton(_("Convert"))
        self.convertButton.enabled = False
        self.layout.addWidget(self.convertButton)

        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.visible = False
        self.layout.addWidget(self.progressBar)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.wordWrap = True
        self.statusLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        self.layout.addWidget(self.statusLabel)
        self.layout.addStretch(1)

        fileButton.connect("clicked()", self.onChooseFile)
        folderButton.connect("clicked()", self.onChooseFolder)
        outputButton.connect("clicked()", self.onChooseOutput)
        self.inputEdit.connect("editingFinished()", self.onInputChanged)
        self.voxelSpinBox.connect("valueChanged(double)", lambda v: self.updateInfo())
        self.levelsSpinBox.connect("valueChanged(int)", lambda v: self.updateInfo())
        self.shardSpinBox.connect("valueChanged(int)", self.onShardChanged)
        self.outputEdit.connect("textChanged(QString)", lambda t: self.updateButtons())
        self.convertButton.connect("clicked()", self.onConvertOrCancel)

    def cleanup(self):
        if self.process is not None:
            self.process.kill()

    # -- input --

    def onChooseFile(self):
        path = qt.QFileDialog.getOpenFileName(
            self.parent, _("Volume to convert"), os.path.dirname(self.inputEdit.text),
            _("Volumes") + " (*.nrrd *.nhdr *.tif *.tiff);;" + _("All files") + " (*)")
        if path:
            self.setInput(path)

    def onChooseFolder(self):
        path = qt.QFileDialog.getExistingDirectory(self.parent, _("Folder of slices, or an OME-Zarr store"),
                                                   os.path.dirname(self.inputEdit.text))
        if path:
            self.setInput(path)

    def onChooseOutput(self):
        path = qt.QFileDialog.getSaveFileName(self.parent, _("OME-Zarr store to write"), self.outputEdit.text,
                                              _("OME-Zarr") + " (*.ome.zarr)")
        if path:
            self.outputEdit.text = path if path.endswith(".zarr") else path + ".ome.zarr"

    def setInput(self, path):
        self.inputEdit.text = path
        self.onInputChanged()

    def onInputChanged(self):
        path = self.inputEdit.text.strip()
        self.info = None
        if path:
            try:
                self.logic.ensureRequirements()
            except Exception as e:  # noqa: BLE001 - shown in the panel
                self.infoLabel.text = str(e)
                self.updateButtons()
                return
            self.outputEdit.text = self.logic.defaultOutput(path)
        self.updateInfo()

    def onShardChanged(self, value):
        rounded = max(128, round(value / 128) * 128)
        if rounded != value:
            self.shardSpinBox.setValue(rounded)
            return
        self.updateInfo()

    def updateInfo(self):
        """Read the input again with the current voxel size and show what will be written."""
        path = self.inputEdit.text.strip()
        if not path or self.process is not None:
            self.updateButtons()
            return
        voxel = self.voxelSpinBox.value or None
        with slicer.util.WaitCursor():
            self.info = self.logic.inspect(path, voxel or 1.0)
        self.voxelSpinBox.enabled = not self.info["voxelFromInput"]
        source = self.info["source"]
        if source is None:
            self.infoLabel.text = self.info["error"]
            self.updateButtons()
            return
        conv = self.logic.converter()
        z, y, x = source.shape
        levels = self.levelsSpinBox.value or conv.auto_levels(source.shape)
        self.levelsSpinBox.setSpecialValueText(_("automatic ({n})").format(n=conv.auto_levels(source.shape)))
        memory = conv.estimated_memory(source.shape, source.dtype, self.shardSpinBox.value, levels)
        size = z * y * x * source.dtype.itemsize
        text = f"{x} × {y} × {z} {source.dtype}, {size / 2**30:.1f} GiB. {source.description}."
        if not self.info["voxelFromInput"] and voxel is None:
            text += " " + _("Set the voxel size.")
        text += " " + _("Conversion needs about {memory:.1f} GiB of memory.").format(memory=memory / 2**30)
        available = self.logic.availableMemory()
        if available and memory > available:
            text += " " + _("Only {available:.1f} GiB are free now: use a smaller shard size or free memory first.").format(
                available=available / 2**30)
        if getattr(source, "needs_native_copy", False):
            text += " " + _("The file's slice order differs from the output's, so level 0 is first copied beside the "
                            "output (temporary, about the size of the compressed level 0).")
        self.infoLabel.text = text
        self.updateButtons()

    def updateButtons(self):
        running = self.process is not None
        ready = bool(self.info and self.info["source"] is not None and self.outputEdit.text.strip()
                     and (self.info["voxelFromInput"] or self.voxelSpinBox.value > 0))
        self.convertButton.text = _("Cancel") if running else _("Convert")
        self.convertButton.enabled = running or ready

    # -- conversion --

    def onConvertOrCancel(self):
        if self.process is not None:
            self.cancel()
        else:
            self.start()

    def start(self):
        inputPath, outputPath = self.inputEdit.text.strip(), os.path.normpath(self.outputEdit.text.strip())
        if os.path.isdir(outputPath) and os.listdir(outputPath):
            if not os.path.exists(os.path.join(outputPath, "zarr.json")):
                slicer.util.errorDisplay(_("{path} exists and is not an OME-Zarr store: choose a new folder.").format(path=outputPath))
                return
            if not slicer.util.confirmOkCancelDisplay(_("Replace the OME-Zarr store {path}?").format(path=outputPath)):
                return
        self.outputWasStore = os.path.exists(os.path.join(outputPath, "zarr.json"))
        python = self.logic.pythonExecutable()
        if not python:
            slicer.util.errorDisplay(_("Slicer's PythonSlicer launcher was not found."))
            return
        voxel = None if self.info["voxelFromInput"] else self.voxelSpinBox.value
        args = self.logic.arguments(inputPath, outputPath, self.levelsSpinBox.value, voxel, self.shardSpinBox.value)
        self.outputPath = outputPath
        self.lastLines = []
        self.process = qt.QProcess()
        self.process.setProcessChannelMode(qt.QProcess.MergedChannels)
        self.process.connect("readyReadStandardOutput()", self.onOutput)
        self.process.connect("finished(int,QProcess::ExitStatus)", self.onFinished)
        self.progressBar.setValue(0)
        self.progressBar.visible = True
        self.statusLabel.text = _("Starting...")
        logging.info(f"OME-Zarr convert: {python} {' '.join(args)}")
        self.process.start(python, args)
        self.updateButtons()

    def onOutput(self):
        if self.process is None:
            return
        while self.process.canReadLine():
            line = bytes(self.process.readLine().data()).decode(errors="replace").rstrip()
            if line.startswith("PROGRESS "):
                _tag, step, done, total = line.split()
                self.progressBar.setValue(int(100 * int(done) / max(1, int(total))))
                self.progressBar.setFormat((_("copying in the file's order: ") if step == "copy" else "") + "%p%")
                continue
            if not line or "warnings.warn" in line or "Warning:" in line:
                continue
            logging.info(f"OME-Zarr convert: {line}")
            self.lastLines = (self.lastLines + [line])[-20:]
            if not line.startswith("DONE"):
                self.statusLabel.text = line.removeprefix("ERROR ")

    def onFinished(self, exitCode, exitStatus):
        self.onOutput()
        process, self.process = self.process, None
        if process is None:
            return
        self.progressBar.visible = False
        self.updateButtons()
        if self.cancelled:
            self.cancelled = False
            return
        if exitStatus != qt.QProcess.NormalExit or exitCode != 0:
            errors = [l.removeprefix("ERROR ") for l in self.lastLines if l.startswith("ERROR ")]
            message = errors[-1] if errors else "\n".join(self.lastLines[-8:])
            self.statusLabel.text = _("Conversion failed: {message}").format(message=message)
            slicer.util.errorDisplay(_("Conversion failed"), detailedText="\n".join(self.lastLines))
            return
        self.statusLabel.text = _("Written and checked: {path}").format(path=self.outputPath)
        if self.openCheckBox.checked:
            with slicer.util.tryWithErrorDisplay(_("Failed to open the store"), waitCursor=True):
                slicer.util.loadNodeFromFile(self.outputPath, "OMEZarr")

    cancelled = False

    def cancel(self):
        if self.process is None:
            return
        self.cancelled = True
        self.process.kill()
        self.process.waitForFinished(10000)
        # Remove what was written: the partial store (a store that was there before is gone already,
        # replaced when the conversion started) and the temporary copy in the file's order.
        for path in (self.outputPath, self.outputPath.rstrip(os.sep) + ".native-copy"):
            if os.path.exists(os.path.join(path, "zarr.json")) or path.endswith(".native-copy"):
                shutil.rmtree(path, ignore_errors=True)
        self.statusLabel.text = _("Cancelled; the partial store was removed.")
        self.progressBar.visible = False
        self.updateButtons()


#
# Test
#


class OMEZarrConvertTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()
        self.tempDir = slicer.util.tempDirectory("OMEZarrConvertTest")
        OMEZarrConvertLogic.ensureRequirements()

    def tearDown(self):
        shutil.rmtree(self.tempDir, ignore_errors=True)

    def runTest(self):
        self.setUp()
        self.test_NrrdRoundTrip()
        self.tearDown()
        self.setUp()
        self.test_SliceFolder()
        self.tearDown()

    def sourceVolume(self, shape=(70, 90, 110), spacing=(0.05, 0.04, 0.03), origin=(12.5, -7.0, 3.25)):
        import numpy as np

        rng = np.random.default_rng(0)
        array = rng.integers(0, 4000, size=shape, dtype=np.uint16)  # k, j, i
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "source")
        slicer.util.updateVolumeFromArray(node, array)
        node.SetSpacing(*spacing)
        node.SetOrigin(*origin)
        return node, array

    def test_NrrdRoundTrip(self):
        """A Slicer-saved NRRD converts, and level 0 opens with the same voxels at the same RAS points."""
        import numpy as np

        self.delayDisplay("NRRD round trip")
        node, array = self.sourceVolume()
        path = os.path.join(self.tempDir, "source.nrrd")
        self.assertTrue(slicer.util.saveNode(node, path, {"useCompression": 1}))
        out = os.path.join(self.tempDir, "source.ome.zarr")
        code, output = OMEZarrConvertLogic.convertBlocking(path, out, levels=2)
        self.assertEqual(code, 0, output)
        self.assertIn("DONE", output)
        loaded = slicer.util.loadNodeFromFile(out, "OMEZarr", {"level": 0, "show": False})
        self.assertIsNotNone(loaded)
        source, result = vtk_matrix(node), vtk_matrix(loaded)
        loadedArray = slicer.util.arrayFromVolume(loaded)
        self.assertEqual(loadedArray.shape, array.shape)
        # Same voxel value at the same RAS point, whatever axis order and flips the store uses.
        rng = np.random.default_rng(1)
        toLoadedIjk = np.linalg.inv(result) @ source
        for _sample in range(200):
            k, j, i = (int(rng.integers(0, n)) for n in array.shape)
            li, lj, lk, _w = np.rint(toLoadedIjk @ [i, j, k, 1]).astype(int)
            self.assertEqual(loadedArray[lk, lj, li], array[k, j, i])
        self.delayDisplay("NRRD round trip passed")

    def test_SliceFolder(self):
        """A folder of TIFF slices with a voxel size converts voxel for voxel."""
        import numpy as np

        self.delayDisplay("Slice folder")
        import tifffile

        rng = np.random.default_rng(2)
        volume = rng.integers(0, 255, size=(40, 50, 60), dtype=np.uint8)
        folder = os.path.join(self.tempDir, "slices")
        os.makedirs(folder)
        for index, image in enumerate(volume):
            tifffile.imwrite(os.path.join(folder, f"slice{index:04d}.tif"), image)
        info = OMEZarrConvertLogic.inspect(folder, 12.0)
        self.assertIsNone(info["error"])
        self.assertFalse(info["voxelFromInput"])
        out = os.path.join(self.tempDir, "slices.ome.zarr")
        code, output = OMEZarrConvertLogic.convertBlocking(folder, out, voxelUm=12.0)
        self.assertEqual(code, 0, output)
        loaded = slicer.util.loadNodeFromFile(out, "OMEZarr", {"level": 0, "show": False})
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(loaded), volume)
        self.assertAlmostEqual(loaded.GetSpacing()[0], 0.012)
        self.delayDisplay("Slice folder passed")


def vtk_matrix(node):
    import numpy as np
    import vtk

    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    return np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
