# Safety Systems Reference

This system drives a ground robot and sprays chemical near people and
crops. Read this before your first field session, even if you already
know the GUI.

If your Husky has a physical wireless E-stop remote (standard
equipment on most A200 units), **that is always the fastest and most
reliable way to stop the robot**, independent of anything below. Use
it without hesitation if you're unsure the software will respond in
time.

---

## The two kinds of E-Stop

The software has **two independent E-stop paths**, and it's important
to understand they're not the same thing:

### 1. Automatic — Husky connection loss
`ROSBridge` treats loss of heartbeat from the Husky's onboard PC as an
E-stop condition automatically — you don't have to do anything for
this to trigger. If the Husky stops sending heartbeats (network drop,
Husky PC crash, radio out of range), the Jetson-side software treats
it as unsafe and stops. This is a deliberate fail-safe design: unknown
state is treated as stopped, not as "probably fine."

### 2. Manual — the GUI E-STOP button
The red **E-STOP** button in the header, and the E-STOP button inside
the Detection tab, both trigger `ActuationController.emergency_stop()`
directly. This:
- Kills all nozzles and the pump immediately
- Sets a flag that makes `actuate()` refuse to fire anything again
  until explicitly cleared (which happens automatically the next time
  you ARM detection — arming builds a fresh controller)

**These two are deliberately kept separate.** Manual E-stop uses its
own flag rather than reusing the automatic one — if it reused the
same flag, the automatic-clear logic (which un-stops the moment the
Husky connection looks healthy again) would silently clear your
manual E-stop within about a tenth of a second, defeating the point
of pressing it.

---

## What happens if the Jetson software crashes or hangs

This is the scenario that matters most and is worth understanding
precisely, because it's the one case where *nothing in the GUI* can
help you — the GUI itself might be the thing that's dead.

**The Arduino firmware has its own independent watchdog.** It does
not trust the Jetson to tell it when something is wrong. If the
Arduino hasn't received *any* command from the Jetson — including the
routine status polls the GUI sends every 1.5 seconds even when idle —
for **4 seconds**, it force-closes the pump and all three nozzles on
its own, in firmware, with no dependency on any software on the
Jetson being alive.

This means: Jetson crash, USB cable pulled, GUI freeze, kernel panic —
all still result in the sprayer shutting itself off within about 4
seconds, because the shutoff logic runs on the Arduino, not the
Jetson.

The Jetson side also detects a dead serial link independently (via
`GantryController`'s own timeout) and will show the connection as
disconnected in the GUI rather than silently displaying stale
"connected" state — but the actual hardware shutoff does not depend
on the Jetson noticing this at all.

---

## Minimum spray hold floor

Once a zone triggers a nozzle, it stays open for a minimum time (0.5s
for weed/herbicide mode, 1.0s for CLS/fungicide mode) even if the
target flickers out of the camera's view immediately after triggering.
This exists because a solenoid valve needs real time to fully open and
deliver a usable dose — without this floor, a target that's only
briefly visible could close the valve before any herbicide actually
reaches it.

This is a **floor, not a fixed burst** — if the target stays in view
longer, the nozzle keeps spraying for as long as it's genuinely
detected. It's not a timer that cuts spraying short.

An E-Stop (either kind, above) always overrides this floor immediately
— safety always wins over guaranteeing a minimum dose.

## Continuous-spray sanity guard

If a nozzle stays continuously open longer than 5 seconds (default),
the system logs a loud warning — but **does not stop spraying**. A
nozzle open that long is either a genuinely large weed patch (spray
is doing its job) or a stuck/false detection silently wasting
chemical. The system can't tell which from software alone, so it
flags it for you to check rather than guessing. Watch the system log
for `⚠ Nozzle N has been continuously open for...` and use your
judgment about what's actually in front of the camera.

## Cross-tab motion lock

Data Collection and Detection each have their own copy of the
navigation controls (a Qt limitation — a widget can't live in two
tabs at once), but they both drive the same physical Husky. Without
coordination, it would be possible to issue a "forward" command from
one tab and a "left" command from the other, with both commands
running on the robot at the same time.

The system prevents this by design: **only one tab's movement
controls are ever enabled at a time.**
- At launch, Data Collection's controls are active; Detection's are
  locked.
- Pressing **ARM DETECTION** unlocks Detection's controls and locks
  Data Collection's — and actively stops anything Data Collection had
  started, so there's no stale motion left running underneath the
  now-locked tab.
- Disarming or E-stopping hands control back to Data Collection.

If a tab's movement buttons look greyed out, this is why — check
which tab is currently armed.

## Non-spray classes (crop protection)

The system will never spray a detection whose class is in
`ZoneConfig.non_spray_classes` (default: `['sugarbeet']`), regardless
of confidence or how long it's been in view. This is enforced at the
earliest possible point — `ZoneManagerRGB._assign_detections()` — so
a non-spray-class detection never even reaches a zone's debounce
counter, let alone triggers a nozzle.

This matters because nothing else in the detection pipeline
distinguishes crop from weed on its own: the detection engine reports
whatever classes the model was trained on, and without this filter,
a correctly-identified crop detection would be treated exactly like a
weed detection and could trigger a spray. The exclusion list exists
specifically to prevent that — spraying herbicide on your own crop.

This is deliberately an **exclude list, not an allow list** of weed
species: a newly retrained model that adds another weed species
should spray-trigger on it by default without any config changes, but
the crop itself must always be excluded, in every mode, with no
exceptions. If you ever need to exclude an additional class (e.g. a
cover crop), add it to `non_spray_classes` — it's a list already.

## Pump/nozzle sequencing

The firmware itself enforces: pump off automatically closes all
nozzles. The GUI mirrors this on the software side too. You cannot
get into a state where nozzles are open with no pump pressure behind
them (or vice versa causing an unexpected spray) through normal
operation.

---

## In an actual emergency

1. **Physical wireless E-stop remote**, if your Husky has one —
   fastest, doesn't depend on any software.
2. **GUI E-STOP button** (header, or inside Detection tab) — stops
   spraying immediately and locks out further automatic spraying
   until you re-arm.
3. If the Jetson is unresponsive and neither of the above seems to be
   working: the Arduino's own 4-second comms watchdog will force
   nozzles and pump off on its own regardless of what the Jetson is
   doing — but don't wait on that as your first response if a person
   or animal is in the spray path right now. Physically disconnect
   power to the pump if you have to.

After any real E-stop event, disarm, review the system log for what
triggered it, and don't re-arm until you understand why it happened.
