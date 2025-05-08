import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

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
    _initialize();
  }

  Future<void> _initialize() async {
    await _initCamera();
    await _loadModel(); 
    _startCameraStream();
  }


  Future<void> _loadModel() async{
    _interpreter = await Interpreter.fromAsset('assets/sawit.tflite');
    print("Input Tensor Shape: ${_interpreter.getInputTensor(0).shape}");
    print("Output Tensor Shape: ${_interpreter.getOutputTensor(0).shape}");
    _interpreter.allocateTensors();
  }

  Future<void> _initCamera() async{
    _controller = CameraController(cameras[0], ResolutionPreset.high);
    await _controller.initialize();
    if (!mounted) return;
    setState(() {});
  }

  /*Float32List imageToByteListFloat32(img.Image image, int inputSize) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        var pixel = image.getPixel(x, y);
        buffer[pixelIndex++] = pixel.r / 255.0;
        buffer[pixelIndex++] = pixel.g / 255.0;
        buffer[pixelIndex++] = pixel.b / 255.0;
      }
    }

    return convertedBytes;
  }*/
  
  Float32List imageToByteListFloat32(img.Image image, int size) {
  var buffer = Float32List(1 * size * size * 3);
  int index = 0;

  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      var pixel = image.getPixel(x, y);

      // Normalize pixel value to 0 - 1
      buffer[index++] = ((pixel >> 16) & 0xFF) / 255.0; // R
      buffer[index++] = ((pixel >> 8) & 0xFF) / 255.0;  // G
      buffer[index++] = (pixel & 0xFF) / 255.0;         // B
    }
  }

  return buffer;
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
    img.Image resizedImage = img.copyResize(oriImage!, width: 800, height: 800);

    Float32List input = imageToByteListFloat32(resizedImage, 800);
    var output = List.filled(1 * 300 * 38, 0.0).reshape([1, 300, 38]);
    print("Input Tensor Length: ${input.length}"); // Harusnya 1 * 800 * 800 * 3 = 1920000
    print("Input Tensor First 10 Values: ${input.sublist(0, 10)}");
        
    try {
      _interpreter.run(input, output);
    } catch (e) {
      print("Error saat menjalankan model: $e");
    }
    
    final endTime = DateTime.now();

    setState(() {
      _isProcessing = false;
      _inferenceTime = endTime.difference(startTime);
      _result = "Deteksi: ${output[0][0].toString()} buah sawit";
    });
    if (_interpreter == null || !_interpreter.isAllocated) {
  print("Interpreter belum diinisialisasi atau tensor belum dialokasikan.");
  return;
    }
  }

Future<void> _pickImageFromGallery() async {
    final picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);

    if (image != null) {
      _runModel(File(image.path));
    }
  }
  
  
  void _startCameraStream() {
  if (!_controller.value.isInitialized) return;

  _controller.startImageStream((CameraImage cameraImage) async {
    if (_isProcessing) return;
    _isProcessing = true;

    try {
      // Konversi frame kamera menjadi img.Image
      img.Image imgFrame = _convertYUV420ToImage(cameraImage);

      // Resize sesuai input tensor
      img.Image resizedImage = img.copyResize(imgFrame, width: 800, height: 800);

      // Konversi menjadi Float32List
      Float32List input = imageToByteListFloat32(resizedImage, 800);

      // Siapkan output tensor
      var output = List.filled(1 * 200 * 200 * 32, 0.0).reshape([1, 200, 200, 32]);

      final startTime = DateTime.now();

      // Inference
      _interpreter.run(input, output);

      //_runModel(input);
      final endTime = DateTime.now();
      final inferenceTime = endTime.difference(startTime);

      setState(() {
        _inferenceTime = inferenceTime;
        _result = "Deteksi: ${output[0][0][0].toString()}";
      });

    } catch (e) {
      print("Error selama inference: $e");
    }

    _isProcessing = false;
  
  });
}
/*
img.Image _convertCameraImage(CameraImage cameraImage) {
  final int width = cameraImage.width;
  final int height = cameraImage.height;
  final img.Image imgImage = img.Image(width, height);

  // Asumsikan format gambar adalah YUV420 (NV21)
  final Plane plane = cameraImage.planes[0];
  final int pixelIndex = 0;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int pixelValue = plane.bytes[y * width + x];

      // Konversi ke RGB (grayscale dalam kasus ini)
      imgImage.setPixel(x, y, img.getColor(pixelValue, pixelValue, pixelValue));
    }
  }

  return imgImage;
}*/

img.Image _convertYUV420ToImage(CameraImage cameraImage) {
  final int width = cameraImage.width;
  final int height = cameraImage.height;

  final img.Image image = img.Image(width, height);

  Plane planeY = cameraImage.planes[0];
  Plane planeU = cameraImage.planes[1];
  Plane planeV = cameraImage.planes[2];

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int pixelIndex = y * width + x;

      int yValue = planeY.bytes[pixelIndex];
      int uValue = planeU.bytes[pixelIndex ~/ 4];
      int vValue = planeV.bytes[pixelIndex ~/ 4];

      double Y = yValue.toDouble();
      double U = uValue.toDouble() - 128.0;
      double V = vValue.toDouble() - 128.0;

      // Konversi YUV ke RGB
      int r = (Y + 1.402 * V).clamp(0, 255).toInt();
      int g = (Y - 0.344136 * U - 0.714136 * V).clamp(0, 255).toInt();
      int b = (Y + 1.772 * U).clamp(0, 255).toInt();

      image.setPixel(x, y, img.getColor(r, g, b));
    }
  }

  return image;
}

void _processCameraImage(CameraImage image, List<List<List<double>>> input) {
    final int width = image.width;
    final int height = image.height;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;
    
    final bytes = image.planes[0].bytes;
    final uvBytes1 = image.planes[1].bytes;
    final uvBytes2 = image.planes[2].bytes;

    // Calculate scaling factors
    final double scaleX = width / 320;
    final double scaleY = height / 320;

    for (int y = 0; y < 800; y++) {
      for (int x = 0; x < 800; x++) {
        final int srcX = (x * scaleX).floor();
        final int srcY = (y * scaleY).floor();
        
        final int uvIndex = 
          uvPixelStride * (srcX >> 1) + 
          uvRowStride * (srcY >> 1);
        
        final int index = srcY * width + srcX;

        final yp = bytes[index];
        final up = uvBytes1[uvIndex];
        final vp = uvBytes2[uvIndex];

        // Optimized YUV to RGB conversion
        int r = yp + ((1436 * (vp - 128)) >> 10);
        int g = yp - ((46549 * (up - 128)) >> 17) - ((93604 * (vp - 128)) >> 17);
        int b = yp + ((1814 * (up - 128)) >> 10);

        // Clamp and normalize
        input[y][x][0] = (r < 0 ? 0 : (r > 255 ? 255 : r)) / 255.0;
        input[y][x][1] = (g < 0 ? 0 : (g > 255 ? 255 : g)) / 255.0;
        input[y][x][2] = (b < 0 ? 0 : (b > 255 ? 255 : b)) / 255.0;
      }
    }}

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
              height: 400,
              width: 800,
              margin: const EdgeInsets.all(50.0),
              child: CameraPreview(_controller),
            )
          else
            Center(child: CircularProgressIndicator()),
          SizedBox(height: 16),
          ElevatedButton(
            onPressed: _pickImageFromGallery,
            child: Text("Ambil dari galeri"),
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