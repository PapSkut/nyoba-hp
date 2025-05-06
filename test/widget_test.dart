import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:nyobahp/main.dart';

void main() {
  testWidgets('App loads and shows UI correctly', (WidgetTester tester) async {
    // Ganti MyApp jadi SawitDetectorApp
    await tester.pumpWidget(SawitDetectorApp());

    // Karena app kamu gak pake counter, kita test yang masuk akal
    expect(find.text('Ambil Foto dan Deteksi'), findsOneWidget);
    expect(find.byType(CameraPreview), findsNothing); // Belum tampil sebelum camera init
  });
}