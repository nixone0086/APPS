#include <stdio.h>
#include <functional>
#include "board.h"
#include "peripherals.h"
#include "pin_mux.h"
#include "clock_config.h"
#include "fsl_debug_console.h"
#include "fsl_gpio.h"
#include "fsl_port.h"
#include "fsl_mrt.h"
#include "mcxn-kit.h"

// **************************************************************
//! System initialization. Do not modify it!!!
void initialize_mcu() __attribute__(( constructor( 0x100 ) ));
void initialize_mcu()
{
    BOARD_InitBootPins();
    BOARD_InitBootClocks();
    BOARD_InitBootPeripherals();
    BOARD_InitDebugConsole();
    CLOCK_EnableClock( kCLOCK_Gpio0 );
    CLOCK_EnableClock( kCLOCK_Gpio1 );
    CLOCK_EnableClock( kCLOCK_Gpio2 );
    CLOCK_EnableClock( kCLOCK_Gpio3 );
    CLOCK_EnableClock( kCLOCK_Gpio4 );
}
DigitalOut led1( P3_16 );
DigitalOut led2( P3_17 );
//! Button instances on MCXN-KIT
DigitalIn button1( P3_18 );
DigitalIn button2( P3_19 );
DigitalIn button3( P3_20 );
DigitalIn button4( P3_21 );

#define T 20

bool are_leds_on = true;

class PWMController
{
public:
    DigitalOut led;
    Ticker ticker;
    uint32_t brightness_level;
    PWMController(pin_name_t pin) : led(pin)
    {
        ticker.attach(&PWMController::control_pwm, this, 1);
        brightness_level = 0;
    }

    void set_brightness(uint8_t level)
    {
        brightness_level = level * T / 255;
    }
protected:
    uint32_t tick_count;
    void control_pwm()
    {
        tick_count++;

        if (this->brightness_level < tick_count) {
            this->led.write(0);
        } else {
            if (are_leds_on) {
                this->led.write(1);
            }
        }

        if (tick_count >= T)
        {
            tick_count = 0;
        }
    }
};
PWMController pwm_leds[] = {
  P3_16,
  P3_17
};
PWMController red_leds[] = {
  P4_00,
  P4_01,
  P4_02,
  P4_03,
  P4_12,
  P4_13,
  P4_16,
  P4_20,
};
void monitor_buttons(void) {
 static bool first_button_pressed = false;
 static bool second_button_pressed = false;

 if (button1.read() == 0 && !first_button_pressed) {
  are_leds_on = false;
  first_button_pressed = true;
 }
 if (button1.read() == 1) {
  first_button_pressed = false;
 }

 if (button2.read() == 0 && !second_button_pressed) {
  are_leds_on = true;
  second_button_pressed = true;
 }
 if (button2.read() == 1) {
  second_button_pressed = false;
 }
}
int main()
{
    red_leds[0].set_brightness(13);
    red_leds[1].set_brightness(32);
    red_leds[2].set_brightness(48);
    red_leds[3].set_brightness(64);
    red_leds[4].set_brightness(80);
    red_leds[5].set_brightness(96);
    red_leds[6].set_brightness(112);
    red_leds[7].set_brightness(128);

    while ( 1 ) {
     monitor_buttons();
     __WFI();
    }
}
