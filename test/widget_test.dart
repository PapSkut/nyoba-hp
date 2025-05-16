import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:nyobahp/main.dart';

void main() {
  testWidgets('App loads and shows UI correctly', (WidgetTester tester) async {
    await tester.pumpWidget(const MaterialApp(home: MyApp()));

    expect(find.text('Deteksi Gallery TFLite'), findsOneWidget);
    expect(find.byType(ElevatedButton), findsOneWidget);
  });
}