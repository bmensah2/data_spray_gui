// ============================================================
//   IMAGING GANTRY — DETECTION CONTROLLER  v1.0
//
//   Purpose:
//     Lightweight firmware for DETECTION MODE only.
//     Camera is fixed at a permanent position — no gantry
//     arm movement required. No homing, no limit switch.
//
//   Controls:
//     - Stepper    : Enabled + holding torque (no movement)
//     - 3x Teejet  : Solenoid nozzle valves (N1/N2/N3)
//     - SHURflo    : Spray pump
//     - AL295W     : Grow light (24V via boost converter)
//     - Servo      : Camera tilt (fixed at boot angle)
//
//   NOT included (use gantry_unified.ino for these):
//     - Stepper motor movement / homing / limit switch
//     - Sequence engine (m, a steps)
//
//   Stepper is ENABLED at boot to lock arm position against
//   vibration and field bumps — but never commanded to move.
//
//   Serial protocol : 9600 baud, '\n' terminated
//   Commands are a strict subset of gantry_unified.ino —
//   same syntax, so the Python GUI works unchanged.
//
//   Pin assignments (IDENTICAL to gantry_unified):
//     Servo           : 6
//     Nozzle 1 relay  : 5      Nozzle 2 relay  : 4
//     Nozzle 3 relay  : 3      Pump relay      : 7
//     Light relay     : 11
//     (Pins 8,9,10,12 unused — stepper/PSU not fitted)
//
//   Author : Nana  |  ABEN PhD Imaging System
// ============================================================

#include <Servo.h>

// ──────────────────────────────────────────────────────────
//  PIN DEFINITIONS (same as gantry_unified for compatibility)
// ──────────────────────────────────────────────────────────
const uint8_t SERVO_PIN     = 6;

// Stepper — enabled at boot to hold torque, no movement
const uint8_t STEP_PIN      = 9;
const uint8_t DIR_PIN       = 8;
const uint8_t ENABLE_PIN    = 10;
const uint8_t MOTOR_PSU_PIN = 12;   // 12V PSU relay

const uint8_t NOZZLE_COUNT  = 3;
const uint8_t RELAY_PIN[3]  = {5, 4, 3};   // N1, N2, N3

const uint8_t PUMP_PIN      = 7;
const uint8_t LIGHT_PIN     = 11;

// ──────────────────────────────────────────────────────────
//  SERVO
// ──────────────────────────────────────────────────────────
const int SERVO_FIXED_ANGLE = 150;   // Fixed detection position
const int SERVO_MIN_ANGLE   = 0;
const int SERVO_MAX_ANGLE   = 170;

Servo camServo;
int   servoAngle = SERVO_FIXED_ANGLE;

// ──────────────────────────────────────────────────────────
//  STATE
// ──────────────────────────────────────────────────────────
bool nozzleState[NOZZLE_COUNT] = {false, false, false};
bool pumpState                 = false;
bool lightState                = false;

// ──────────────────────────────────────────────────────────
//  COMMS WATCHDOG
//
//  The Jetson polls "p" every 1.5s (see gantry_controller.py
//  POLL_INTERVAL) even when idle, so ANY complete command
//  (not just nozzle/pump commands) counts as a live heartbeat.
//
//  If no command is received for COMM_TIMEOUT_MS, we assume the
//  Jetson has crashed, hung, or the USB/serial link has died and
//  force pump + all nozzles OFF regardless of last commanded
//  state. This is the last line of defense — it does NOT depend
//  on any software on the host being alive.
// ──────────────────────────────────────────────────────────
const unsigned long COMM_TIMEOUT_MS = 4000;   // ~2.6x the 1.5s poll interval
unsigned long        lastCmdMillis  = 0;
bool                  watchdogTripped = false;

// ──────────────────────────────────────────────────────────
//  SERIAL BUFFER (no String — zero heap allocation)
// ──────────────────────────────────────────────────────────
const uint8_t CMD_BUF_SIZE = 64;
char    cmdBuf[CMD_BUF_SIZE];
uint8_t cmdLen   = 0;
bool    cmdReady = false;

// ──────────────────────────────────────────────────────────
//  FORWARD DECLARATIONS
// ──────────────────────────────────────────────────────────
void processCommand(char* cmd);
void setNozzle(uint8_t index, bool on);
void allNozzlesOn();
void allNozzlesOff();
void setPump(bool on);
void setLight(bool on);
void setServoAngle(int angle);
bool validateServoRange(int angle);
void checkCommsWatchdog();
void printFullStatus();
void printBanner();
void printMenu();

// ──────────────────────────────────────────────────────────
//  SETUP
// ──────────────────────────────────────────────────────────
void setup() {
  // Stepper — enable hold torque to lock arm position
  // Motor PSU relay ON first to power the driver
  pinMode(MOTOR_PSU_PIN, OUTPUT);
  digitalWrite(MOTOR_PSU_PIN, LOW);    // active LOW — PSU ON

  // Enable pin LOW = driver active, holding current applied
  pinMode(STEP_PIN,   OUTPUT);
  pinMode(DIR_PIN,    OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);       // hold torque — arm locked

  // Servo — fixed at detection angle
  camServo.attach(SERVO_PIN);
  camServo.write(SERVO_FIXED_ANGLE);

  // Nozzle relays — all OFF (active LOW)
  for (uint8_t i = 0; i < NOZZLE_COUNT; i++) {
    pinMode(RELAY_PIN[i], OUTPUT);
    digitalWrite(RELAY_PIN[i], HIGH);
  }

  // Pump relay — OFF
  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, HIGH);

  // Light relay — OFF
  pinMode(LIGHT_PIN, OUTPUT);
  digitalWrite(LIGHT_PIN, HIGH);

  Serial.begin(9600);
  lastCmdMillis = millis();
  printBanner();
  printMenu();
}

// ──────────────────────────────────────────────────────────
//  MAIN LOOP — serial command processing only
// ──────────────────────────────────────────────────────────
void loop() {
  // Byte-by-byte serial intake — no String allocation
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdLen > 0) {
        cmdBuf[cmdLen] = '\0';
        cmdReady = true;
      }
    } else {
      if (cmdLen < CMD_BUF_SIZE - 1) {
        cmdBuf[cmdLen++] = c;
      }
    }
  }

  if (cmdReady) {
    // Any complete line — including routine "p" status polls —
    // counts as proof the Jetson is alive and talking to us.
    lastCmdMillis   = millis();
    watchdogTripped = false;

    processCommand(cmdBuf);
    cmdLen   = 0;
    cmdReady = false;
  }

  checkCommsWatchdog();
}

// ──────────────────────────────────────────────────────────
//  COMMS WATCHDOG CHECK
//  Call every loop() pass. Uses subtraction so it is safe
//  across the ~49-day millis() rollover.
// ──────────────────────────────────────────────────────────
void checkCommsWatchdog() {
  unsigned long elapsed = millis() - lastCmdMillis;

  if (elapsed > COMM_TIMEOUT_MS && !watchdogTripped) {
    watchdogTripped = true;
    Serial.println(F("[WDT] Comm link lost — forcing pump + nozzles OFF"));
    allNozzlesOff();
    setPump(false);
  }
}

// ──────────────────────────────────────────────────────────
//  COMMAND PROCESSOR
//  Supports the same syntax as gantry_unified.ino
//  so Python GUI works unchanged.
// ──────────────────────────────────────────────────────────
void processCommand(char* cmd) {
  // Trim leading whitespace
  while (*cmd == ' ') cmd++;
  if (*cmd == '\0') return;

  // ── PUMP ────────────────────────────────────────────────
  if (strcmp(cmd, "pump on")  == 0 || strcmp(cmd, "pon")  == 0) {
    setPump(true);  return;
  }
  if (strcmp(cmd, "pump off") == 0 || strcmp(cmd, "poff") == 0) {
    setPump(false); return;
  }

  // ── NOZZLES ─────────────────────────────────────────────
  if (strcmp(cmd, "na on")  == 0) { allNozzlesOn();  return; }
  if (strcmp(cmd, "na off") == 0) { allNozzlesOff(); return; }

  // n1 on/off, n2 on/off, n3 on/off
  if (cmd[0] == 'n' && cmd[1] >= '1' && cmd[1] <= '3') {
    uint8_t idx = cmd[1] - '1';       // 0-based
    bool    on  = (strstr(cmd + 2, "on") != nullptr);
    setNozzle(idx, on);
    return;
  }

  // ── LIGHT ───────────────────────────────────────────────
  if (strcmp(cmd, "light on")  == 0 || strcmp(cmd, "lon")  == 0) {
    setLight(true);  return;
  }
  if (strcmp(cmd, "light off") == 0 || strcmp(cmd, "loff") == 0) {
    setLight(false); return;
  }

  // ── SERVO ───────────────────────────────────────────────
  // a <deg> — adjust camera angle even in detection mode
  if (cmd[0] == 'a' && cmd[1] == ' ') {
    int angle = atoi(cmd + 2);
    if (validateServoRange(angle)) {
      setServoAngle(angle);
    }
    return;
  }

  // ── STATUS / INFO ────────────────────────────────────────
  if (strcmp(cmd, "p")    == 0) { printFullStatus(); return; }
  if (strcmp(cmd, "i")    == 0) { printFullStatus(); return; }
  if (strcmp(cmd, "help") == 0 ||
      strcmp(cmd, "?")    == 0) { printMenu();       return; }

  // ── GRACEFULLY IGNORE GANTRY COMMANDS ────────────────────
  // h, m, r, s, hs, e, d, mpsu, loop, pause, resume, stop
  // These are valid in gantry_unified but silently ignored here
  // so the GUI doesn't error — it just gets no response
  if (cmd[0] == 'h' || cmd[0] == 'm' || cmd[0] == 'r' ||
      cmd[0] == 's' || cmd[0] == 'e' || cmd[0] == 'd') {
    Serial.println(F("[INFO] Stepper commands ignored in detection mode"));
    Serial.println(F("       Upload gantry_unified.ino for full gantry control"));
    return;
  }
  if (strncmp(cmd, "mpsu",  4) == 0 ||
      strncmp(cmd, "loop",  4) == 0 ||
      strncmp(cmd, "pause", 5) == 0 ||
      strncmp(cmd, "resume",6) == 0 ||
      strncmp(cmd, "stop",  4) == 0) {
    Serial.println(F("[INFO] Command not available in detection mode"));
    return;
  }

  Serial.print(F("[?] Unknown command: "));
  Serial.println(cmd);
}

// ──────────────────────────────────────────────────────────
//  HARDWARE CONTROL
// ──────────────────────────────────────────────────────────
void setNozzle(uint8_t index, bool on) {
  if (index >= NOZZLE_COUNT) return;
  nozzleState[index] = on;
  digitalWrite(RELAY_PIN[index], on ? LOW : HIGH);  // active LOW
  // Match gantry_controller.py [NZ] parse token
  Serial.print(F("[NZ] Nozzle ")); Serial.print(index + 1);
  Serial.println(on ? F(" ON") : F(" OFF"));
}

void allNozzlesOn() {
  for (uint8_t i = 0; i < NOZZLE_COUNT; i++) setNozzle(i, true);
}

void allNozzlesOff() {
  for (uint8_t i = 0; i < NOZZLE_COUNT; i++) setNozzle(i, false);
}

void setPump(bool on) {
  if (!on) allNozzlesOff();         // safety: nozzles off before pump off
  pumpState = on;
  digitalWrite(PUMP_PIN, on ? LOW : HIGH);  // active LOW
  // Match gantry_controller.py parse tokens
  Serial.println(on ? F("[PMP] Pump ON") : F("[PMP] Pump OFF"));
}

void setLight(bool on) {
  lightState = on;
  digitalWrite(LIGHT_PIN, on ? LOW : HIGH);  // active LOW
  // Match gantry_controller.py [LT] parse token
  Serial.println(on ? F("[LT] Light ON") : F("[LT] Light OFF"));
}

void setServoAngle(int angle) {
  servoAngle = angle;
  camServo.write(angle);
  Serial.print(F("[OK] SERVO ")); Serial.print(angle);
  Serial.println(F(" deg"));
}

bool validateServoRange(int angle) {
  if (angle < SERVO_MIN_ANGLE || angle > SERVO_MAX_ANGLE) {
    Serial.print(F("[!] Angle out of range (0-"));
    Serial.print(SERVO_MAX_ANGLE); Serial.println(F(" deg)"));
    return false;
  }
  return true;
}

// ──────────────────────────────────────────────────────────
//  STATUS
// ──────────────────────────────────────────────────────────
void printFullStatus() {
  Serial.println(F("    ======= DETECTION STATUS ======="));
  Serial.println(F("    [MODE] DETECTION"));
  Serial.print(F("    Mode     : DETECTION (camera fixed at "));
  Serial.print(SERVO_FIXED_ANGLE); Serial.println(F(" deg)"));
  Serial.println(F("    Motor PSU: ON  [POWERED]"));
  Serial.print(F("    Cam angle: ")); Serial.print(servoAngle);
  Serial.println(F(" deg"));
  Serial.println(F("    ---------------------------"));
  Serial.print(F("    Pump     : "));
  Serial.println(pumpState ? F("ON  [RUNNING]") : F("OFF [STOPPED]"));
  for (uint8_t i = 0; i < NOZZLE_COUNT; i++) {
    Serial.print(F("    Nozzle ")); Serial.print(i + 1);
    Serial.print(F("  : "));
    Serial.println(nozzleState[i] ? F("ON  [OPEN]") : F("OFF [CLOSED]"));
  }
  Serial.println(F("    ---------------------------"));
  Serial.print(F("    Light    : "));
  Serial.println(lightState ? F("ON  [ACTIVE]") : F("OFF [STANDBY]"));
  Serial.println(F("    ---------------------------"));
  Serial.println(F("    Stepper  : ENABLED [HOLDING TORQUE]"));
  Serial.println(F("    Motor PSU: ON  [POWERED]"));
  Serial.println(F("    Homing   : N/A (detection mode)"));
  Serial.print(F("    Comm WDT : "));
  Serial.print(millis() - lastCmdMillis);
  Serial.print(F("ms since last cmd (timeout "));
  Serial.print(COMM_TIMEOUT_MS);
  Serial.println(F("ms)"));
  Serial.println(F("    ==========================="));
}

void printBanner() {
  Serial.println(F("============================================"));
  Serial.println(F("   ABEN GANTRY — DETECTION MODE  v1.0"));
  Serial.println(F("   Stepper HOLD | Nozzles | Pump | Light | Servo"));
  Serial.println(F("   Camera fixed — arm locked, no movement"));
  Serial.println(F("============================================"));
  Serial.println(F("Pins:"));
  Serial.println(F("  Stepper  STEP=9 DIR=8 EN=10 PSU=12 [HOLD]"));
  Serial.println(F("  Servo    6  (fixed at 145 deg)"));
  Serial.println(F("  Nozzles  N1=5  N2=4  N3=3"));
  Serial.println(F("  Pump     7"));
  Serial.println(F("  Light    11"));
  Serial.println();
}

void printMenu() {
  Serial.println(F("\n======== DETECTION MODE COMMANDS ========"));
  Serial.println(F("PUMP:"));
  Serial.println(F("  pump on / pump off    pon / poff"));
  Serial.println(F(""));
  Serial.println(F("NOZZLES:"));
  Serial.println(F("  n1 on/off  n2 on/off  n3 on/off"));
  Serial.println(F("  na on / na off  (all nozzles)"));
  Serial.println(F(""));
  Serial.println(F("LIGHT:"));
  Serial.println(F("  light on / light off  lon / loff"));
  Serial.println(F(""));
  Serial.println(F("SERVO (fine adjust only):"));
  Serial.println(F("  a <deg>   (0-170, default 145)"));
  Serial.println(F(""));
  Serial.println(F("STATUS:"));
  Serial.println(F("  p / i     - Full status"));
  Serial.println(F("  help / ?  - This menu"));
  Serial.println(F(""));
  Serial.println(F("NOTE: Stepper commands (h, m, r, s, e, d)"));
  Serial.println(F("      are not available in detection mode."));
  Serial.println(F("      Use gantry_unified.ino for full control."));
  Serial.println(F("=========================================\n"));
}
