#include <Arduino.h>

void setup() {
    pinMode(0, OUTPUT);
    pinMode(1, OUTPUT);
}

void loop() {
    // blocking
    digitalWrite(1,HIGH);
    delayMicroseconds(10);
    digitalWrite(1,LOW);
    delayMicroseconds(10);
}
