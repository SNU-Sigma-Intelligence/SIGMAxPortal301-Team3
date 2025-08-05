#include <SoftwareSerial.h>
SoftwareSerial lidarSerial(10, 11); // RX, TX

void sendCommand(byte* cmd, int len) {
  for (int i = 0; i < len; i++) {
    lidarSerial.write(cmd[i]);
  }
}

void setup() {
  Serial.begin(9600);
  lidarSerial.begin(115200);

  delay(500); // 안정화

  // Mode B 측정 시작 (반복 측정 + 레이저 유지)
  byte cmd[] = { 0xCD, 0x01, 0x06, 0x07 };
  sendCommand(cmd, sizeof(cmd));
}

void loop() {
  // 1. 버퍼에 데이터가 있으면 계속 읽기
  // Serial.println(lidarSerial.available());
  while (lidarSerial.available()) {
    byte header = lidarSerial.read();
    Serial.print("0x");
    Serial.println(header, HEX);

    // 2. 헤더(0xFA)를 찾았을 때만 처리 시작
    if (header == 0xFA) {
      byte buf[8];  // 나머지 8바이트를 여기에 저장
      buf[0] = header;
      int bytesRead = 1;

      // 3. 나머지 8바이트가 모두 들어올 때까지 대기
      unsigned long start = millis();
      while (bytesRead < 8 && millis() - start < 100) {
        if (lidarSerial.available()) {
          buf[bytesRead++] = lidarSerial.read();
        }
      }

      // 4. 충분히 다 읽었을 때만 처리
      if (bytesRead == 8) {
        // 패킷 형태: 0xFA 0x00 0x05 0xZZ 0xYY 0x00 0x00 Cksum
        if (buf[2] == 0x05) {
          int distance = buf[3] | (buf[4] << 8);  // little-endian
          Serial.print("Distance: ");
          Serial.print(distance);
          Serial.println(" mm");
        } else {
          Serial.println("[WARN] Invalid packet type.");
        }
      } else {
        Serial.println("[WARN] Incomplete packet.");
      }
    }
  }
}
