#include <SoftwareSerial.h>
SoftwareSerial lidarSerial(10, 11); // RX, TX

const int PACKET_SIZE = 15; // Mode C 응답 패킷 크기
byte recvBuffer[PACKET_SIZE];
int recvIndex = 0;
bool receiving = false;

void sendModeC0Command() {
  byte cmd[15];
  cmd[0] = 0xCD;
  cmd[1] = 0x04;
  cmd[2] = 0x00; // Mode C0

  // Epoch time in milliseconds (big-endian, 8바이트)
  unsigned long ms = millis();
  uint64_t epoch = (uint64_t)ms;
  for (int i = 0; i < 8; i++) {
    cmd[3 + i] = (epoch >> (8 * (7 - i))) & 0xFF;
  }

  // 남은 바이트 채우기 (빈 공간 3바이트 + 체크섬)
  for (int i = 11; i < 14; i++) cmd[i] = 0x00;

  // 체크섬: Byte 1~13 합계
  byte checksum = 0;
  for (int i = 1; i < 14; i++) {
    checksum += cmd[i];
  }
  cmd[14] = checksum;

  // 명령 전송
  for (int i = 0; i < 15; i++) {
    lidarSerial.write(cmd[i]);
  }

  Serial.println("[CMD] Mode C0 start command sent");
}

void setup() {
  Serial.begin(9600);
  lidarSerial.begin(115200);
  delay(500);

  Serial.println("setup start");

  sendModeC0Command();
  Serial.println("setup finished");
}

void loop() {
  while (lidarSerial.available()) {
    byte b = lidarSerial.read();

    if (!receiving) {
      if (b == 0xFA) {
        recvBuffer[0] = b;
        recvIndex = 1;
        receiving = true;
      }
    } else {
      recvBuffer[recvIndex++] = b;
      if (recvIndex == PACKET_SIZE) {
        // 패킷 수신 완료
        receiving = false;
        recvIndex = 0;

        // 유효성 검사
        if (recvBuffer[2] == 0x0D) {
          int distance = recvBuffer[4] << 8 | recvBuffer[3];
          if (distance == 0xFFFF) {
            Serial.println("[WARN] Measurement failed");
          } else {
            Serial.print("Distance: ");
            Serial.print(distance);
            Serial.println(" mm");
          }
        } else {
          Serial.println("[WARN] Invalid Mode C packet");
        }
      }
    }
  }
}
