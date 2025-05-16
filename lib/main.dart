import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'dart:math' as math;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  bool isDarkMode = false;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mock iPalms',
      debugShowCheckedModeBanner: false,
      theme: isDarkMode ? ThemeData.dark() : ThemeData.light(),
      home: GalleryDetectionPage(
        isDarkMode: isDarkMode,
        onToggleTheme: () {
          setState(() {
            isDarkMode = !isDarkMode;
          });
        },
      ),
    );
  }
}

class GalleryDetectionPage extends StatefulWidget {
  final bool isDarkMode;
  final VoidCallback onToggleTheme;

  const GalleryDetectionPage({
    super.key,
    required this.isDarkMode,
    required this.onToggleTheme,
  });

  @override
  State<GalleryDetectionPage> createState() => _GalleryDetectionPageState();
}

class _GalleryDetectionPageState extends State<GalleryDetectionPage> {
  Interpreter? _interpreter;
  List<int>? _inputShape;
  List<int>? _outputShape;
  File? _imageFile;
  img.Image? _imageRaw;
  List<Map<String, dynamic>> _detections = [];
  bool _loading = false;
  String? _error;
  double? _elapsed;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      final interpreter = await Interpreter.fromAsset('assets/sawit.tflite');
      interpreter.allocateTensors();
      setState(() {
        _interpreter = interpreter;
        _inputShape = interpreter.getInputTensor(0).shape;
        _outputShape = interpreter.getOutputTensor(0).shape;
        _error = null;
      });
    } catch (e) {
      setState(() {
        _error = 'Gagal load model: $e';
      });
    }
  }

  Future<void> _pickAndDetect() async {
    setState(() {
      _loading = true;
      _detections = [];
      _error = null;
    });
    try {
      final picked = await ImagePicker().pickImage(source: ImageSource.gallery);
      if (picked == null) {
        setState(() => _loading = false);
        return;
      }
      final file = File(picked.path);
      setState(() => _imageFile = file);
      await _runDetection(file);
    } catch (e) {
      setState(() {
        _loading = false;
        _error = 'Gagal ambil gambar: $e';
      });
    }
  }

  Future<void> _captureAndDetect() async {
    setState(() {
      _loading = true;
      _detections = [];
      _error = null;
    });
    try {
      final picked = await ImagePicker().pickImage(source: ImageSource.camera);
      if (picked == null) {
        setState(() => _loading = false);
        return;
      }
      final file = File(picked.path);
      setState(() => _imageFile = file);
      await _runDetection(file);
    } catch (e) {
      setState(() {
        _loading = false;
        _error = 'Gagal ambil gambar: $e';
      });
    }
  }

  Future<void> _runDetection(File file) async {
    if (_interpreter == null || _inputShape == null || _outputShape == null) {
      setState(() {
        _loading = false;
        _error = 'Model belum siap';
      });
      return;
    }
    try {
      final start = DateTime.now();
      final bytes = await file.readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) throw Exception('Gagal decode gambar');

      final inputH = _inputShape![1];
      final inputW = _inputShape![2];
      final resized = img.copyResize(
        decoded,
        width: inputW,
        height: inputH,
        interpolation: img.Interpolation.linear,
      );
      setState(() => _imageRaw = resized);

      final input = [
        List.generate(
          inputH,
          (y) => List.generate(inputW, (x) {
            final pixel = resized.getPixel(x, y);
            return [
              img.getRed(pixel) / 255.0,
              img.getGreen(pixel) / 255.0,
              img.getBlue(pixel) / 255.0,
            ];
          }),
        ),
      ];

      final output = List.generate(
        1,
        (_) => List.generate(300, (_) => List<double>.filled(38, 0.0)),
      );

      final maskOutput = List.generate(
        1,
        (_) => List.generate(
          200,
          (_) => List.generate(200, (_) => List<double>.filled(32, 0.0)),
        ),
      );

      final outputs = {0: output, 1: maskOutput};
      _interpreter!.runForMultipleInputs([input], outputs);

      final dets = _parseDetections(output[0]);
      final end = DateTime.now();
      setState(() {
        _detections = dets;
        _loading = false;
        _elapsed = end.difference(start).inMilliseconds / 1000.0;
        _error = null;
      });
    } catch (e) {
      setState(() {
        _loading = false;
        _error = 'Error deteksi: $e';
        _elapsed = null;
      });
    }
  }

  List<Map<String, dynamic>> _parseDetections(List<List<double>> output) {
    final results = <Map<String, dynamic>>[];
    for (final det in output) {
      if (det.length >= 5 && det[4] > 0.5) {
        final bbox = [
          det[0].clamp(0.0, 1.0),
          det[1].clamp(0.0, 1.0),
          det[2].clamp(0.0, 1.0),
          det[3].clamp(0.0, 1.0),
        ];
        results.add({'bbox': bbox, 'confidence': det[4]});
      }
    }
    return nonMaxSuppression(results, 0.5);
  }

  List<Map<String, dynamic>> nonMaxSuppression(
    List<Map<String, dynamic>> dets,
    double iouThresh,
  ) {
    dets.sort(
      (a, b) =>
          (b['confidence'] as double).compareTo(a['confidence'] as double),
    );
    final kept = <Map<String, dynamic>>[];
    while (dets.isNotEmpty) {
      final best = dets.removeAt(0);
      kept.add(best);
      dets.removeWhere((det) {
        final iou = computeIoU(best['bbox'], det['bbox']);
        return iou > iouThresh;
      });
    }
    return kept;
  }

  double computeIoU(List bbox1, List bbox2) {
    final x1 = math.max(bbox1[0], bbox2[0]);
    final y1 = math.max(bbox1[1], bbox2[1]);
    final x2 = math.min(bbox1[2], bbox2[2]);
    final y2 = math.min(bbox1[3], bbox2[3]);
    final interArea = math.max(0, x2 - x1) * math.max(0, y2 - y1);
    final area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    final area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
    return interArea / (area1 + area2 - interArea);
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  Widget _buildInfoBox(String label, String value) {
    return Container(
      width: 260,
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
      margin: const EdgeInsets.symmetric(vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.6),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        '$label: $value',
        textAlign: TextAlign.center,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 18,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final imageWidget =
        _imageFile == null
            ? const Center(child: Text('Pilih Gambar'))
            : LayoutBuilder(
              builder: (context, constraints) {
                final displayWidth = constraints.maxWidth;
                final displayHeight = constraints.maxHeight;
                return Stack(
                  children: [
                    if (_imageRaw != null)
                      Center(
                        child: Image.memory(
                          Uint8List.fromList(img.encodeJpg(_imageRaw!)),
                          width: displayWidth,
                          height: displayHeight,
                          fit: BoxFit.contain,
                        ),
                      ),
                    if (_detections.isNotEmpty)
                      CustomPaint(
                        painter: _DetectionPainter(
                          detections: _detections,
                          imageSize: const Size(800, 800),
                          widgetSize: Size(displayWidth, displayHeight),
                        ),
                        size: Size(displayWidth, displayHeight),
                      ),
                  ],
                );
              },
            );

    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text('Mock iPalms'),
        actions: [
          IconButton(
            icon: Icon(widget.isDarkMode ? Icons.light_mode : Icons.dark_mode),
            onPressed: widget.onToggleTheme,
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child:
                _loading
                    ? const Center(child: CircularProgressIndicator())
                    : imageWidget,
          ),
          if (_detections.isNotEmpty || _elapsed != null)
            Column(
              children: [
                _buildInfoBox('Jumlah Sawit', _detections.length.toString()),
                if (_elapsed != null)
                  _buildInfoBox(
                    'Waktu Deteksi',
                    '${_elapsed!.toStringAsFixed(2)} detik',
                  ),
              ],
            ),
          if (_error != null)
            Padding(
              padding: const EdgeInsets.all(8),
              child: Text(_error!, style: const TextStyle(color: Colors.red)),
            ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton.icon(
                  onPressed:
                      _loading || _interpreter == null ? null : _pickAndDetect,
                  icon: const Icon(Icons.image),
                  label: const Text('Pilih Gambar'),
                ),
                const SizedBox(width: 16),
                ElevatedButton.icon(
                  onPressed:
                      _loading || _interpreter == null
                          ? null
                          : _captureAndDetect,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Kamera'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _DetectionPainter extends CustomPainter {
  final List detections;
  final Size imageSize;
  final Size widgetSize;

  _DetectionPainter({
    required this.detections,
    required this.imageSize,
    required this.widgetSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint =
        Paint()
          ..color = Colors.yellow
          ..strokeWidth = 2
          ..style = PaintingStyle.stroke;

    final imgW = imageSize.width;
    final imgH = imageSize.height;
    final widgetW = widgetSize.width;
    final widgetH = widgetSize.height;
    final scale = math.min(widgetW / imgW, widgetH / imgH);
    final displayW = imgW * scale;
    final displayH = imgH * scale;
    final dx = (widgetW - displayW) / 2;
    final dy = (widgetH - displayH) / 2;

    for (final det in detections) {
      final bbox = det['bbox'] as List;
      final conf = det['confidence'] as double;
      final x1 = bbox[0] * imgW * scale + dx;
      final y1 = bbox[1] * imgH * scale + dy;
      final x2 = bbox[2] * imgW * scale + dx;
      final y2 = bbox[3] * imgH * scale + dy;
      final rect = Rect.fromLTRB(x1, y1, x2, y2);
      canvas.drawRect(rect, paint);

      final textPainter = TextPainter(
        text: TextSpan(
          text: '${(conf * 100).toStringAsFixed(0)}%',
          style: const TextStyle(
            color: Colors.yellow,
            fontSize: 14,
            backgroundColor: Colors.black54,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(canvas, Offset(x1, y1 - 20));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}