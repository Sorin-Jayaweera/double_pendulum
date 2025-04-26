/********
E80 Lab 7 Dive Activity Code
Authors:
    Omar Aleman (oaleman@g.hmc.edu) '21 (contributed 2019)
    Wilson Ives (wives@g.hmc.edu) '20 (contributed in 2018)
    Christopher McElroy (cmcelroy@g.hmc.edu) '19 (contributed in 2017)  
    Josephine Wong (jowong@hmc.edu) '18 (contributed in 2016)
    Apoorva Sharma (asharma@hmc.edu) '17 (contributed in 2016)                    
*/

//#include "GravityTDS.h"
#include <SPI.h>
#include <SD.h>

#define pressurePin 14 //A0
#define turbidityPin 15 // A1
#define phPin 16 // A2
#define temperaturePin 17 //A3
#define salinityPin A10 //A10

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

#include <EEPROM.h>
int filenum;
int filenumAddress = 15; //for EEPROM, 15 is a random value, I don't want it to interfere with whatever EEPROM stuff the automatic code does
char namebuffer[10];
bool resetFileCounter = false; //TODO: Make True whenever we take out all data and restart the count. This is once a day maybe.

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

void setup() {
  

  logger.include(&imu);
  logger.include(&gps);
  logger.include(&xy_state_estimator);
  logger.include(&z_state_estimator);
  logger.include(&depth_control);
  logger.include(&motor_driver);
  logger.include(&adc);
  logger.include(&ef);
  logger.include(&button_sampler);
  logger.init();

  printer.init();
  ef.init();
  button_sampler.init();
  imu.init();
  UartSerial.begin(9600);
  gps.init(&GPS);
  motor_driver.init();
  led.init();

  int delayUntilStart = 30000;
  int diveDelay = 3000; // how long robot will stay at depth waypoint before continuing (ms)

  const int num_depth_waypoints = 8;
  double depth_waypoints [] = {0.2,0.4,0.6,0.8,1,1.2,1.4,1.8,0 };  // listed as z0,z1,... etc.
  depth_control.init(num_depth_waypoints, depth_waypoints, diveDelay);
  
  xy_state_estimator.init(); 
  z_state_estimator.init();

  printer.printMessage("Starting main loop",10);
  loopStartTime = millis();
  printer.lastExecutionTime            = loopStartTime - LOOP_PERIOD + PRINTER_LOOP_OFFSET ;
  imu.lastExecutionTime                = loopStartTime - LOOP_PERIOD + IMU_LOOP_OFFSET;
  adc.lastExecutionTime                = loopStartTime - LOOP_PERIOD + ADC_LOOP_OFFSET;
  ef.lastExecutionTime                 = loopStartTime - LOOP_PERIOD + ERROR_FLAG_LOOP_OFFSET;
  button_sampler.lastExecutionTime     = loopStartTime - LOOP_PERIOD + BUTTON_LOOP_OFFSET;
  xy_state_estimator.lastExecutionTime = loopStartTime - LOOP_PERIOD + XY_STATE_ESTIMATOR_LOOP_OFFSET;
  z_state_estimator.lastExecutionTime  = loopStartTime - LOOP_PERIOD + Z_STATE_ESTIMATOR_LOOP_OFFSET;
  depth_control.lastExecutionTime      = loopStartTime - LOOP_PERIOD + DEPTH_CONTROL_LOOP_OFFSET;
  logger.lastExecutionTime             = loopStartTime - LOOP_PERIOD + LOGGER_LOOP_OFFSET;

  pinMode(pressurePin,INPUT);
  pinMode(turbidityPin,INPUT);
  pinMode(phPin,INPUT);
  pinMode(temperaturePin,INPUT);
  pinMode(salinityPin,INPUT);

  Serial.print("Initializing SD card...");


  if(resetFileCounter){
    resetFileCounter = false;
    filenum = 0;
  }
  else{
    filenum = EEPROM.read(filenumAddress);
    filenum += 1;
  }
  EEPROM.write(filenumAddress,filenum);
  
  //TODO: What is the CS pin for SD?
 while (!SD.begin(10)) { //CS = 10
   Serial.println("SD initialization failed!");
 }
  Serial.println("initialization done.");
  sprintf(namebuffer,"log%d.csv",filenum);
  Serial.print("File nameBuffer: "); Serial.println(namebuffer);
  


  delay(delayUntilStart);
  Serial.println("Delay done");
}



//////////////////////////////* Loop */////////////////////////

bool firstCycle = true;
void loop() {
  currentTime=millis();
  float bittovolt = 3.3/1023;

  if(firstCycle){
    firstCycle = false;
    
    File filelog = SD.open(namebuffer,FILE_WRITE);
    filelog.println("Pressure, Turbidity, PH, Temperature, Salinity");
    filelog.close();
    Serial.println();

    Serial.print("Pressure, Turbidity, PH, Temperature, Salinity,Time,uV");
    
  }

  // moved to be next to logger.log so that things are synced element wise
  
  // float turbidityVoltage = analogRead(turbidityPin) * bittovolt;
  // float phVoltage = analogRead(phPin)* bittovolt;
  // float depthVoltage = analogRead(pressurePin)* bittovolt;
  // float temperatureVoltage = analogRead(temperaturePin)* bittovolt;
  // float tdsVoltage = analogRead(salinityPin)* bittovolt;

  // Serial.print(depthVoltage); Serial.print(",");
  // Serial.print(turbidityVoltage); Serial.print(",");
  // Serial.print(phVoltage); Serial.print(",");
  // Serial.print(temperatureVoltage); Serial.print(",");
  // Serial.print(tdsVoltage); Serial.print(",");
  // Serial.println(millis()); Serial.println();

  //  File file = SD.open("log1.csv",FILE_WRITE);
  //  file.print(depthVoltage); file.print(",");
  //  file.print(turbidityVoltage); file.print(",");
  //  file.print(phVoltage); file.print(",");
  //  file.print(temperatureVoltage); file.print(",");
  //  file.print(tdsVoltage); file.print(",");
  //  file.print(millis());file.println();
  //  file.close();
  
    if ( currentTime-printer.lastExecutionTime > LOOP_PERIOD ) {
      printer.lastExecutionTime = currentTime;
      printer.printValue(0,adc.printSample());
      //printer.printValue(1,ef.printStates());
      printer.printValue(1,button_sampler.printState());
      printer.printValue(2,logger.printState());
      printer.printValue(3,gps.printState());   
      printer.printValue(4,xy_state_estimator.printState());  
      printer.printValue(5,z_state_estimator.printState());  
      printer.printValue(6,depth_control.printWaypointUpdate());
      printer.printValue(7,depth_control.printString());
      printer.printValue(8,motor_driver.printState());
      printer.printValue(9,imu.printRollPitchHeading());        
      printer.printValue(10,imu.printAccels());
      printer.printToSerial();  // To stop printing, just comment this line out
   }

  /* ROBOT CONTROL Finite State Machine */
  if ( currentTime-depth_control.lastExecutionTime > LOOP_PERIOD ) {
    depth_control.lastExecutionTime = currentTime;
    if ( depth_control.diveState ) {      // DIVE STATE //
      depth_control.complete = false;
      if ( !depth_control.atDepth ) {
        depth_control.dive(&z_state_estimator.state, currentTime);
      }
      else {
        depth_control.diveState = false; 
        depth_control.surfaceState = true;
      }
      motor_driver.drive(depth_control.uV,depth_control.uV,depth_control.uV);
  
    }
    if ( depth_control.surfaceState ) {     // SURFACE STATE //
      if ( !depth_control.atSurface ) { 
        depth_control.surface(&z_state_estimator.state);
      }
      else if ( depth_control.complete ) { 
        delete[] depth_control.wayPoints;   // destroy depth waypoint array from the Heap
      }
      motor_driver.drive(depth_control.uV,depth_control.uV,depth_control.uV);
    }
  }
  
  if ( currentTime-adc.lastExecutionTime > LOOP_PERIOD ) {
    adc.lastExecutionTime = currentTime;
    adc.updateSample(); 
  }

  if ( currentTime-ef.lastExecutionTime > LOOP_PERIOD ) {
    ef.lastExecutionTime = currentTime;
    attachInterrupt(digitalPinToInterrupt(ERROR_FLAG_A), EFA_Detected, LOW);
    attachInterrupt(digitalPinToInterrupt(ERROR_FLAG_B), EFB_Detected, LOW);
    attachInterrupt(digitalPinToInterrupt(ERROR_FLAG_C), EFC_Detected, LOW);
    delay(5);
    detachInterrupt(digitalPinToInterrupt(ERROR_FLAG_A));
    detachInterrupt(digitalPinToInterrupt(ERROR_FLAG_B));
    detachInterrupt(digitalPinToInterrupt(ERROR_FLAG_C));
    ef.updateStates(EF_States[0],EF_States[1],EF_States[2]);
    EF_States[0] = 1;
    EF_States[1] = 1;
    EF_States[2] = 1;
  }

  if ( currentTime-button_sampler.lastExecutionTime > LOOP_PERIOD ) {
    button_sampler.lastExecutionTime = currentTime;
    button_sampler.updateState();
  }

  if ( currentTime-imu.lastExecutionTime > LOOP_PERIOD ) {
    imu.lastExecutionTime = currentTime;
    imu.read();     // blocking I2C calls
  }
 
  gps.read(&GPS); // blocking UART calls, need to check for UART data every cycle

  if ( currentTime-xy_state_estimator.lastExecutionTime > LOOP_PERIOD ) {
    xy_state_estimator.lastExecutionTime = currentTime;
    xy_state_estimator.updateState(&imu.state, &gps.state);
  }

  if ( currentTime-z_state_estimator.lastExecutionTime > LOOP_PERIOD ) {
    z_state_estimator.lastExecutionTime = currentTime;
    z_state_estimator.updateState(analogRead(PRESSURE_PIN));
  }
  
  if ( currentTime-led.lastExecutionTime > LOOP_PERIOD ) {
    led.lastExecutionTime = currentTime;
    led.flashLED(&gps.state);
  }

  if ( currentTime- logger.lastExecutionTime > LOOP_PERIOD && logger.keepLogging ) {
    logger.lastExecutionTime = currentTime;
    logger.log();

    
    float turbidityVoltage = analogRead(turbidityPin) * bittovolt;
    float phVoltage = analogRead(phPin)* bittovolt;
    float depthVoltage = analogRead(pressurePin)* bittovolt;
    float temperatureVoltage = analogRead(temperaturePin)* bittovolt;
    float tdsVoltage = analogRead(salinityPin)* bittovolt;

    // Serial.print(depthVoltage); Serial.print(",");
    // Serial.print(turbidityVoltage); Serial.print(",");
    // Serial.print(phVoltage); Serial.print(",");
    // Serial.print(temperatureVoltage); Serial.print(",");
    // Serial.print(tdsVoltage); Serial.print(",");
    // Serial.println(millis()); Serial.println();

    File filelog = SD.open(namebuffer,FILE_WRITE);
    filelog.print(depthVoltage); filelog.print(",");
    filelog.print(turbidityVoltage); filelog.print(",");
    filelog.print(phVoltage); filelog.print(",");
    filelog.print(temperatureVoltage); filelog.print(",");
    filelog.print(tdsVoltage); filelog.print(",");
    filelog.print(millis());filelog.print(",");
    filelog.print(depth_control.uV);filelog.println();
    filelog.close();
  }
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
