
#include <Arduino.h>
#include <Wire.h>
#include <avr/io.h>
#include <avr/interrupt.h>

#include <Pinouts.h>
#include <TimingOffsets.h>
#include <SensorGPS.h>
#include <SensorIMU.h>
#include <XYStateEstimator.h>
#include <ZStateEstimator.h>
#include <ADCSampler.h>
#include <ErrorFlagSampler.h>
#include <ButtonSampler.h> // A template of a data source library
#include <MotorDriver.h>
#include <Logger.h>
#include <Printer.h>
#include <DepthControl.h>
#define UartSerial Serial1
#include <GPSLockLED.h>

/////////////////////////* Global Variables *////////////////////////

MotorDriver motor_driver;
XYStateEstimator xy_state_estimator;
ZStateEstimator z_state_estimator;
DepthControl depth_control;
SensorGPS gps;
Adafruit_GPS GPS(&UartSerial);
ADCSampler adc;
ErrorFlagSampler ef;
ButtonSampler button_sampler;
SensorIMU imu;
Logger logger;
Printer printer;
GPSLockLED led;

// loop start recorder
int loopStartTime;
int currentTime;
volatile bool EF_States[NUM_FLAGS] = {1,1,1};

////////////////////////* Setup *////////////////////////////////

int lasttime;

void setup() {
  Serial.begin(115200);
  Serial.println("start");
  lasttime = millis();
}



//////////////////////////////* Loop */////////////////////////
int control = -1;
void loop() {
Serial.println("loop");
  int uv = 125;
  if(millis() - 5000 > lasttime){
    lasttime = millis();
    control = -1 * control;
  }
  
  motor_driver.drive(control * uv, control * uv,control * uv);

}

void EFA_Detected(void){
  EF_States[0] = 0;
}

void EFB_Detected(void){
  EF_States[1] = 0;
}

void EFC_Detected(void){
  EF_States[2] = 0;
}
