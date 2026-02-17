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
// ******************
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
// ******************
//! Global data
//! LEDs on MCXN-KIT - instances of class DigitalOut
DigitalOut g_led_P3_16(P3_16);
DigitalOut g_led_P3_17(P3_17);
//! Button on MCXN-KIT - instance of class DigitalIn
DigitalIn g_but_P3_18(P3_18);
DigitalIn g_but_P3_19(P3_19);
DigitalIn g_but_P3_20(P3_20);
DigitalIn g_but_P3_21(P3_21);
//! Callback function for LED control
#define T 20
class PWMLED
{
public:
 DigitalOut m_ledka;
 Ticker m_ticker;
 uint32_t m_T0;
 PWMLED(pin_name_t t_pin) :
   m_ledka(t_pin), m_T0(0), _m_ticks(0)
 {
  m_ticker.attach(&PWMLED::pwm, this, 1);
 }
 void nastav_jas(uint8_t t_proc)
 {
  if (t_proc > 100)
   t_proc = 100;
  m_T0 = (static_cast<uint32_t>(t_proc) * T) / 100;
 }
protected:
 uint32_t _m_ticks;
 void pwm() {
  if (_m_ticks < m_T0) {
   m_ledka.write(1);
  }
  else {
   m_ledka.write(0);
  }
  _m_ticks++;
  if (_m_ticks >= T)
  {
   _m_ticks = 0;
  }
 }
};
PWMLED g_leds[] = {
		P3_16,
		P3_17
};
PWMLED red_leds[] = {
		P4_00,
		P4_01,
		P4_02,
		P4_03,
		P4_12,
		P4_13,
		P4_16,
		P4_20
};
PWMLED rgb1[] = {
		P0_14,
		P0_15,
		P0_22 };
PWMLED rgb2[] = {
		P0_24,
		P0_25,
		P0_26
};
PWMLED rgb3[] = {
		P0_28,
		P0_29,
		P0_30
};
float rgb1_barva[] = { 255, 0, 0 };
float rgb2_barva[] = { 0, 128, 0 };
float rgb3_barva[] = { 255, 255, 0 };

int pocet_kliku = 0;
int pocet = 0;
bool stop = false;
bool buttonPressed = false;
int stavsemaforu = 0;


int main()
{

 while (1) {
int a =0;
	 if(g_but_P3_19.read() == 0 && g_but_P3_20.read() == 0){
	 		rgb1[0].nastav_jas(100);
	 		rgb2[0].nastav_jas(100);
	 		while(a<10){
	 		rgb3[2].nastav_jas(100);
	 		rgb3[2].nastav_jas(0);
	 		a++;
	 		}
	 	}else{
	 		rgb1[0].nastav_jas(0);
	 		rgb2[0].nastav_jas(0);
	 		rgb3[2].nastav_jas(0);
	 	}

if (g_but_P3_18.read() == 0) {
   while (stop != true) {
    if (stop == false) {
     for (int i = 0; i < 8; i++) {
      delay_ms(100);
      for (int j = 0; j <= i; j++) {
       if (j == i) {
        red_leds[j].nastav_jas(100);
       }
       else {
        red_leds[j].nastav_jas(5);
       }
      }
      if (pocet > 3) {
       pocet = 0;
      } else {
       pocet++;
      }
     }
     for (int i = 0; i < 8; i++) {
      delay_ms(100);
      if (i == 0) {
       red_leds[7].nastav_jas(5);
      }
      red_leds[i].nastav_jas(0);
     }
         }
         for (int i = 0; i < 3; i++) {
          if (pocet == 0) {
           rgb1[i].nastav_jas(rgb1_barva[i]);
           rgb2[i].nastav_jas(0);
           rgb3[i].nastav_jas(0);
          }
          else if (pocet == 1) {
           rgb2[i].nastav_jas(rgb2_barva[i]);
           rgb1[i].nastav_jas(0);
           rgb3[i].nastav_jas(0);
          } else if (pocet == 2) {
           rgb3[i].nastav_jas(rgb3_barva[i]);
           rgb1[i].nastav_jas(0);
           rgb2[i].nastav_jas(0);
          }
         }
        }
//        if(g_but_P3_19.read() == 0){
//             delay_ms(200);
//             pocet = 0;
//             delay_ms(200);
//            }

       }
      }
     //     if(g_but_P3_18.read() == 0){
     //      g_leds[0].nastav_jas(100);
     //      g_leds[1].nastav_jas(0);
     //     }
     //     if(g_but_P3_19.read() == 0) {
     //      g_leds[0].nastav_jas(0);
     //         g_leds[1].nastav_jas(100);
     //     }
      while (1)
       __WFI();
     }
