#include <SoftwareSerial.h>

class DRB222{
public:
  enum Error_Type{
    ERROR_NONE,
    ERROR_CHKSUM,
    ERROR_CMD_NOT_FOUND,
    ERROR_DIST_OUT_OF_RANGE,
    ERROR_CMD_OUT_OF_RANGE,
    ERROR_NOT_TURNED_ON,
    ERROR_LOW_SNR,
    ERROR_WRONG_HEADER,
    ERROR_WRONG_HEADER_RECVED
  };
  enum Measure_Mode{
    MODE_A,
    MODE_B,
    MODE_C0,
    MODE_C1,
    MODE_C2,
    MODE_D
  };

  DRB222(int rxPin, int txPin) : serial(rxPin, txPin){}

  bool init(Measure_Mode mode = MODE_A){
    serial.begin(115200);

    this->mode = mode;
    if(mode == MODE_A){
      uint8_t cmd_on[2] = { 0x01, 0x03 };
      sendPacket(cmd_on, 2);
      uint8_t recvBuffer[256];
      uint16_t len = recvPacket(recvBuffer);
      if(len != 0) return false;
    }
    if(mode == MODE_C0 || mode == MODE_C1 || mode == MODE_C2){
      uint8_t cmd_on[2] = { 0x04, };
      if(mode == MODE_C0) cmd_on[1] = 0x00;
      else if(mode == MODE_C1) cmd_on[1] = 0x01;
      else if(mode == MODE_C2) cmd_on[1] = 0x02;
      sendPacket(cmd_on, 2);
      uint8_t recvBuffer[256];
      int len = recvPacket(recvBuffer);
      if(len != 0) return false;
    }
    if(mode == MODE_D){
      uint8_t cmd_on[2] = { 0x01, 0x0A };
      sendPacket(cmd_on, 2);
      uint8_t recvBuffer[256];
      int len = recvPacket(recvBuffer);
      if(len != 0) return false;
    }
    return true;
  }

  uint32_t measure(){
    uint8_t recvBuffer[256];
    if(mode == MODE_A){
      uint8_t cmd_measure[2] = { 0x01, 0x05 };
      sendPacket(cmd_measure, 2);
      uint16_t len = recvPacket(recvBuffer);
      if(len != 4) return -1;
      return recvBuffer[0] | (recvBuffer[1] << 8);
    }
    else if(mode == MODE_B){
      uint8_t cmd_measure[2] = { 0x01, 0x06 };
      sendPacket(cmd_measure, 2);
      uint16_t len = recvPacket(recvBuffer);
      if(len != 4) return -1;
      return recvBuffer[0] | (recvBuffer[1] << 8);
    }
    else if(mode == MODE_C0 || mode == MODE_C1 || mode == MODE_C2){
      uint16_t len = recvPacket(recvBuffer);
      if(len != 12) return -1;
      epochTime = *(uint64_t *)(recvBuffer + 4);
      return recvBuffer[0] | (recvBuffer[1] << 8);
    }
    else{
      uint16_t len = recvPacket(recvBuffer);
      if(len != 4) return -1;
      return recvBuffer[0] | (recvBuffer[1] << 8);
    }
  }

  Error_Type getLastError(){
    return error;
  }

  uint64_t getEpochTime(){
    return epochTime;
  }


private:
  uint16_t recvPacket(uint8_t *buf, unsigned int timeout = 1000){
    unsigned int start = millis();
    while(millis() < start + timeout){
      if(!serial.available()) continue;
      uint8_t header = serial.read();
      if(header != 0xFA && header != 0x0E){
        error = ERROR_WRONG_HEADER_RECVED;
        continue;
      }
      if(header == 0xFA){
        uint8_t checksum = 0;
        uint16_t length = (serial.read() << 8) | (serial.read());
        if(length > 30){
          error = ERROR_WRONG_HEADER;
          return -1;
        }
        checksum += (length & 0xff);
        checksum += ((length >> 8) & 0xff);
        for(int i = 0; i < length - 1; i++){
          buf[i] = serial.read();
          checksum += buf[i];
        }
        if(checksum != serial.read()){
          error = ERROR_CHKSUM;
          return -1;
        }
        return length - 1;
      } else if(header == 0x0E){
        uint8_t error_code = serial.read();
        switch(error_code){
        case 0x81:
          error = ERROR_CHKSUM;
          break;
        case 0x82:
          error = ERROR_CMD_NOT_FOUND;
          break;
        case 0x83:
          error = ERROR_DIST_OUT_OF_RANGE;
          break;
        case 0x84:
          error = ERROR_CMD_OUT_OF_RANGE;
          break;
        case 0x85:
          error = ERROR_NOT_TURNED_ON;
          break;
        case 0x89:
          error = ERROR_LOW_SNR;
          break;
        case 0x8B:
          error = ERROR_WRONG_HEADER;
          break;
        }
        return -1;
      }
    }
    return -1;
  }

  void sendPacket(uint8_t* buf, uint16_t len){
    uint8_t checksum = 0;
    for(int i = 0; i < len; i++) checksum += buf[i];

    serial.write(0xCD);
    serial.write(buf, len);
    serial.write(checksum);
  }

  SoftwareSerial serial;
  Measure_Mode mode;
  Error_Type error = ERROR_NONE;

  uint64_t epochTime;
};

DRB222 lidar(10, 11);

void setup() {
  Serial.begin(9600);
  if(lidar.init(DRB222::MODE_B)){
    Serial.println("initialized");
  } else{
    Serial.println("intitialization failed");
  }
}

void loop() {
  int distance = lidar.measure();
  Serial.print("distance(mm): ");
  Serial.println(distance);
  DRB222::Error_Type error = lidar.getLastError();
  if(error != DRB222::ERROR_NONE){
    Serial.println(error);
  }
  delay(1000);
}
