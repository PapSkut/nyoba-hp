import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;

List<CameraDescription> cameras = [];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(SawitDetectorApp());
}

class SawitDetectorApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late CameraController _controller;
  late Interpreter _interpreter;
  bool _isProcessing = false;
  String _result = "";
  Duration? _inferenceTime;

  @override
  void initState() {
    super.initState();
    _loadModel();
    _initCamera();
  }

  void _initCamera() async {
    _controller = CameraController(cameras[0], ResolutionPreset.medium);
    await _controller.initialize();
    if (!mounted) return;
    setState(() {});
  }

  void _loadModel() async {
    _interpreter = await Interpreter.fromAsset('model.tflite');
  }

  Float32List imageToByteListFloat32(img.Image image, int inputSize) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        int pixel = image.getPixel(x, y) as int;
        buffer[pixelIndex++] = ((pixel >> 16) & 0xFF) / 255.0;
        buffer[pixelIndex++] = ((pixel >> 8) & 0xFF) / 255.0;
        buffer[pixelIndex++] = (pixel & 0xFF) / 255.0;
      }
    }

    return convertedBytes;
  }

  void _runModel(File imageFile) async {
    setState(() {
      _isProcessing = true;
      _result = "";
      _inferenceTime = null;
    });

    final startTime = DateTime.now();

    final rawBytes = await imageFile.readAsBytes();
    img.Image? oriImage = img.decodeImage(rawBytes);
    img.Image resizedImage = img.copyResize(oriImage!, width: 224, height: 224);

    Float32List input = imageToByteListFloat32(resizedImage, 224);
    var output = List.filled(1 * 1, 0).reshape([1, 1]);

    _interpreter.run(input, output);

    final endTime = DateTime.now();

    setState(() {
      _isProcessing = false;
      _inferenceTime = endTime.difference(startTime);
      _result = "Deteksi: ${output[0][0].toString()} buah sawit";
    });
  }

  Future<void> _takePicture() async {
    if (!_controller.value.isInitialized || _isProcessing) return;

    await _controller.takePicture().then((XFile? file) {
      if (file != null) {
        _runModel(File(file.path));
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Sawit Detector")),
      body: Column(
        children: [
          if (_controller.value.isInitialized)
            Container(
              height: 300,
              child: CameraPreview(_controller),
            )
          else
            Center(child: CircularProgressIndicator()),
          SizedBox(height: 16),
          ElevatedButton(
            onPressed: _takePicture,
            child: Text("Coba Foto Daw"),
          ),
          SizedBox(height: 16),
          if (_isProcessing) CircularProgressIndicator(),
          if (_result.isNotEmpty) Text(_result, style: TextStyle(fontSize: 18)),
          if (_inferenceTime != null)
            Text("Waktu Deteksi: ${_inferenceTime!.inMilliseconds} ms")
        ],
      ),
    );
  }
}