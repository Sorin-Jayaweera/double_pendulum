#include <EEPROM.h>

int addr = 15; // address of filenum name
int val = 0; // so that we start filenum from zero

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.println("EEPROM WRITER");
  EEPROM.write(addr,val);
  Serial.print("written addr "); Serial.print(addr); Serial.print(" with val: "); Serial.println(val);

}

void loop() {
  // put your main code here, to run repeatedly:

}
