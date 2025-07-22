#include <Wire.h>
#include <VL53L0X_modified.h>

#define N_SENSORS 4
const int xshutPins[N_SENSORS] = {2, 3, 4, 5};
const uint8_t addresses[N_SENSORS] = {0x30, 0x31, 0x32, 0x33};
int sensor_amt = 4;

VL53L0X sensors[N_SENSORS];

void initSensors() {
  for (int i = 0; i < sensor_amt; i++) {
    pinMode(xshutPins[i], OUTPUT);
    digitalWrite(xshutPins[i], LOW);
  }
  delay(100);

  for (int i = 0; i < sensor_amt; i++) {
    digitalWrite(xshutPins[i], HIGH);
    delay(10);
    sensors[i].init(true);
    sensors[i].setAddress(addresses[i]);
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  initSensors();
}

void loop() {
  // for (int i = 0; i < sensor_amt; i++) {
  //   sensors[i].readyRangeSingle();
  // }
  for (int i = 0; i < sensor_amt; i++) {
    uint16_t dist = sensors[i].readRangeSingleMillimeters();
    Serial.print(dist);
    if (i < sensor_amt - 1) Serial.print(',');
  }
  Serial.println();
}
