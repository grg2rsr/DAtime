#include <Arduino.h>

int PIN_1 = 2;
int PIN_2 = 1;

// params
const bool ramp = false; // sets mode
float pin_1_ramp_up_time = 500.0; // in us
float pin_1_ramp_down_time = 1000.0; // in us
float pin_2_ramp_up_time = 500.0; // in us
float pin_2_ramp_down_time = 1000.0; // in us

// inferred
float pin_1_m_up = 4095 / pin_1_ramp_up_time;
float pin_1_m_down = 4095 / pin_1_ramp_down_time;
float pin_2_m_up = 4095 / pin_2_ramp_up_time;
float pin_2_m_down = 4095 / pin_2_ramp_down_time;

// declarations
int v_min = 0;
int v_max = 4095; // 2**12 - 1

int pin_1_v;
bool pin_1_is_high = false;
bool pin_1_state = false;
long pin_1_t_on = 0;
long pin_1_t_off = 0;
bool pin_1_ramp_up_done = false;
bool pin_1_ramp_down_done = false;

int pin_2_v;
bool pin_2_is_high = false;
bool pin_2_state = false;
long pin_2_t_on = 0;
long pin_2_t_off = 0;
bool pin_2_ramp_up_done = false;
bool pin_2_ramp_down_done = false;

long dt;

bool state = false; // for debug pin

void read_pins(){
  // pin_1
  pin_1_state = digitalRead(PIN_1);

  // onset
  if (pin_1_state == true && pin_1_is_high == false){
    pin_1_is_high = true;
    pin_1_t_on = micros();
    digitalWrite(13, HIGH);
  }
  
  // offset
  if (pin_1_state == false && pin_1_is_high == true){
    pin_1_is_high = false;
    pin_1_t_off = micros();
    digitalWrite(13, LOW);
  }

  // pin_2
  pin_2_state = digitalRead(PIN_2);

  // onset
  if (pin_2_state == true && pin_2_is_high == false){
    pin_2_is_high = true;
    pin_2_t_on = micros();
    digitalWrite(13, HIGH);
  }
  
  // offset
  if (pin_2_state == false && pin_2_is_high == true){
    pin_2_is_high = false;
    pin_2_t_off = micros();
    digitalWrite(13, LOW);
  }
}

void calc_voltages_ramp(){
  // calculate voltages

  // pin_1
  // ramping up 
  if (pin_1_is_high == true && pin_1_ramp_up_done == false){
    dt = micros() - pin_1_t_on;
    pin_1_v = pin_1_m_up * dt;
    if (pin_1_v > v_max){
      pin_1_v = v_max;
      pin_1_ramp_up_done = true;
      pin_1_ramp_down_done = false;
    }
  }

  // ramping down
  if (pin_1_is_high == false && pin_1_ramp_down_done == false){
    dt = micros() - pin_1_t_off;
    pin_1_v = v_max - (pin_1_m_down * dt);
    if (pin_1_v < v_min){
      pin_1_v = v_min;
      pin_1_ramp_down_done = true;
      pin_1_ramp_up_done = false;
    }
  }

  // pin_2
  // ramping up 
  if (pin_2_is_high == true && pin_2_ramp_up_done == false){
    dt = micros() - pin_2_t_on;
    pin_2_v = pin_2_m_up * dt;
    if (pin_2_v > v_max){
      pin_2_v = v_max;
      pin_2_ramp_up_done = true;
      pin_2_ramp_down_done = false;
    }
  }

  // ramping down
  if (pin_2_is_high == false && pin_2_ramp_down_done == false){
    dt = micros() - pin_2_t_off;
    pin_2_v = v_max - (pin_2_m_down * dt);
    if (pin_2_v < v_min){
      pin_2_v = v_min;
      pin_2_ramp_down_done = true;
      pin_2_ramp_up_done = false;
    }
  }
}

void calc_voltages_step(){
  // calculate voltages

  // pin_1
  // ramping up 
  if (pin_1_is_high == true){
    pin_1_v = v_max;
  }

  // ramping down
  if (pin_1_is_high == false){
    pin_1_v = v_min;
  }

  // pin_2
  // ramping up 
  if (pin_2_is_high == true){
    pin_2_v = v_max;
  }

  // ramping down
  if (pin_2_is_high == false){
    pin_2_v = v_min;
  }
}

void set_voltages(){
    analogWriteDAC0(pin_1_v);
    analogWriteDAC1(pin_2_v);
}

int PIN_T = 3;

void setup() {
    pinMode(PIN_1, INPUT);
    pinMode(PIN_2, INPUT);

    // debug pin
    pinMode(0, OUTPUT);
    pinMode(PIN_T, OUTPUT);

    // led vis pin
    pinMode(13, OUTPUT);

    // initialize dac
    analogWriteResolution(12);
}

// unsigned long t_last_stim = 0;
// unsigned long iti = 10000;
// bool pin_t_state = false;
// unsigned long pulse_dur = 2000;
// unsigned long t_last_pulse = 0;

// unsigned long t_last_stim_on = 0;
// unsigned long isi = 100000;

// int n_pulses = 5;
// unsigned long ipi = 10000;
// unsigned long pulse_dur = 1000;

// bool pin_t_state = false;
// unsigned long t_last_pulse_on = 0;
// unsigned long now = 0;

void loop() {
  // read digital pins
  read_pins();

  // fire pins by itself
  // now = micros();
  // if (now - t_last_stim_on > isi){
  //   t_last_stim_on = now;
    
  // }

  // if (micros() - t_last_pulse > pulse_dur && pin_t_state == true){
  //   pin_t_state = false;
  //   digitalWrite(PIN_T, LOW);
  // }

  // // fire pins by itself
  // if (micros() - t_last_stim > iti && pin_t_state == false){
  //   t_last_stim = micros();
  //   pin_t_state = true;
  //   t_last_pulse = micros();
  //   digitalWrite(PIN_T, HIGH);
  // }

  // if (micros() - t_last_pulse > pulse_dur && pin_t_state == true){
  //   pin_t_state = false;
  //   digitalWrite(PIN_T, LOW);
  // }


  // calculates voltages
  if (ramp == true){
    calc_voltages_step();
  }
  else{
    calc_voltages_ramp();
  }

  // sets them
  set_voltages();

  // speed debug
//   if (state == true){
//       digitalWrite(0, HIGH);
//   }
//   else {
//       digitalWrite(0, LOW);
//   }
//   state = !state;
}
