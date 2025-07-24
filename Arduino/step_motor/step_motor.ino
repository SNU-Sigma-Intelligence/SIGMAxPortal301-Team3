#include <Stepper.h>

 

const int stepsPerRevolution = 200;

 

Stepper myStepper1(stepsPerRevolution, 3, 4, 5, 6);
Stepper myStepper2(stepsPerRevolution, 8, 9, 10, 11);

 

void setup() {

  myStepper1.setSpeed(60);
  myStepper2.setSpeed(60);
  //myStepper2.setSpeed(60);

  Serial.begin(9600);

}

 

void loop() {
  Serial.println("clockwise");
  for(int i=1;i<=stepsPerRevolution;i++) {
    myStepper1.step(1);
    myStepper2.step(2);
    delay(100);
  }
    delay(2000);
  Serial.println("counterclockwise");
  for(int i=1;i<=stepsPerRevolution;i++) {
    myStepper1.step(-1);
    myStepper2.step(-2);
    delay(100);
  }
    delay(2000);
}