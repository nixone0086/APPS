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
#include "fsl_lpspi.h"
#include "mcxn-kit.h"
#include "lcd_lib.h"
#include "cts_lib.h"
#include "graph_class.hpp"

void _mcu_initialization() __attribute__((constructor( 0x100 )));

void _mcu_initialization() {
    BOARD_InitBootPins();
    BOARD_InitBootClocks();
    BOARD_InitBootPeripherals();
    BOARD_InitDebugConsole();
    CLOCK_EnableClock(kCLOCK_Gpio0);
    CLOCK_EnableClock(kCLOCK_Gpio1);
    CLOCK_EnableClock(kCLOCK_Gpio2);
    CLOCK_EnableClock(kCLOCK_Gpio3);
    CLOCK_EnableClock(kCLOCK_Gpio4);
}

DigitalOut g_led_P3_16(P3_16);
DigitalOut g_led_P3_17(P3_17);
DigitalIn g_but_P3_18(P3_18);
DigitalIn g_but_P3_19(P3_19);
DigitalIn g_but_P3_20(P3_20);
DigitalIn g_but_P3_21(P3_21);


RGB red = {255, 0, 0};
RGB green = {0, 255, 0};
RGB blue = {0, 0, 255};
RGB white = {255, 255, 255};
RGB black = {0, 0, 0};
RGB selected_color = red;

void set_red_color() { selected_color = red; }
void set_green_color() { selected_color = green; }
void set_blue_color() { selected_color = blue; }
void set_white_color() { selected_color = white; }


class Triangle : public GraphElement {
public:
    Point2D m_p1, m_p2, m_p3;

    Triangle(Point2D t_p1, Point2D t_p2, Point2D t_p3, RGB t_fg, RGB t_bg)
        : GraphElement(t_fg, t_bg), m_p1(t_p1), m_p2(t_p2), m_p3(t_p3) {}

    void draw() override {
        // Draw the triangle edges using lines
        Line line1(m_p1, m_p2, m_fg_color, m_bg_color);
        Line line2(m_p2, m_p3, m_fg_color, m_bg_color);
        Line line3(m_p3, m_p1, m_fg_color, m_bg_color);
        line1.draw();
        line2.draw();
        line3.draw();
    }

    // Method to fill the triangle with the selected color
    void fill_shape(RGB fill_color) {
        int minX = std::min({m_p1.x, m_p2.x, m_p3.x});
        int maxX = std::max({m_p1.x, m_p2.x, m_p3.x});
        int minY = std::min({m_p1.y, m_p2.y, m_p3.y});
        int maxY = std::max({m_p1.y, m_p2.y, m_p3.y});

        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                if (is_point_inside_triangle({x, y})) {
                    // Fill the pixel inside the triangle with the selected color
                    lcd_put_pixel(x, y, ((fill_color.r >> 3) << 11) | ((fill_color.g >> 2) << 5) | (fill_color.b >> 3));
                }
            }
        }
    }

    bool is_point_inside_triangle(Point2D p) {

        float area = abs((m_p1.x * (m_p2.y - m_p3.y) +
                          m_p2.x * (m_p3.y - m_p1.y) +
                          m_p3.x * (m_p1.y - m_p2.y)) / 2.0);


        float area1 = abs((p.x * (m_p2.y - m_p3.y) +
                          m_p2.x * (m_p3.y - p.y) +
                          m_p3.x * (p.y - m_p2.y)) / 2.0);

        float area2 = abs((m_p1.x * (p.y - m_p3.y) +
                          p.x * (m_p3.y - m_p1.y) +
                          m_p3.x * (m_p1.y - p.y)) / 2.0);

        float area3 = abs((m_p1.x * (m_p2.y - p.y) +
                          m_p2.x * (p.y - m_p1.y) +
                          p.x * (m_p1.y - m_p2.y)) / 2.0);

        return (abs(area - (area1 + area2 + area3)) < 0.1);
    }
};


class Ellipse : public GraphElement {
public:
    Point2D m_center;
    int32_t m_a, m_b;

    Ellipse(Point2D t_center, int32_t t_a, int32_t t_b, RGB t_fg, RGB t_bg)
        : GraphElement(t_fg, t_bg), m_center(t_center), m_a(t_a), m_b(t_b) {}

    void draw() override {
        int x = 0, y = m_b;
        int a2 = m_a * m_a, b2 = m_b * m_b;
        int dx = 0, dy = 2 * a2 * y;
        int err = b2 - (2 * b2 * m_a) + a2;

        while (x <= m_a) {
            lcd_put_pixel(m_center.x + x-1, m_center.y + y-1, ((m_fg_color.r >> 3) << 11) | ((m_fg_color.g >> 2) << 5) | (m_fg_color.b >> 3));
            lcd_put_pixel(m_center.x - x+1, m_center.y + y-1, ((m_fg_color.r >> 3) << 11) | ((m_fg_color.g >> 2) << 5) | (m_fg_color.b >> 3));
            lcd_put_pixel(m_center.x + x-1, m_center.y - y+1, ((m_fg_color.r >> 3) << 11) | ((m_fg_color.g >> 2) << 5) | (m_fg_color.b >> 3));
            lcd_put_pixel(m_center.x - x+1, m_center.y - y+1, ((m_fg_color.r >> 3) << 11) | ((m_fg_color.g >> 2) << 5) | (m_fg_color.b >> 3));

            if (err >= 0) {
                y--;
                dy -= 2 * b2;
                err -= dy;
            }

            x++;
            dx += 2 * a2;
            err += dx + a2;
        }
    }
    // Method to fill the ellipse with the selected color
    void fill_shape(RGB fill_color) {
        int minX = m_center.x - m_a;
        int maxX = m_center.x + m_a;
        int minY = m_center.y - m_b;
        int maxY = m_center.y + m_b;

        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                if (is_point_inside_ellipse({x, y})) {
                    lcd_put_pixel(x, y, ((fill_color.r >> 3) << 11) | ((fill_color.g >> 2) << 5) | (fill_color.b >> 3));
                }
            }
        }
    }


    bool is_point_inside_ellipse(Point2D p) {
        int dx = p.x - m_center.x;
        int dy = p.y - m_center.y;

        return (dx * dx * m_b * m_b + dy * dy * m_a * m_a <= m_a * m_a * m_b * m_b);
    }
};

void delay_ms(int ms) {
    for (volatile int i = 0; i < ms * 5000; i++) {}
}

int main() {
    PRINTF("LCD demo program started...\n");

    lcd_init();

    if (cts_init() < 0) {
        PRINTF("Touch Screen not detected!\n");
    }

    int button_size = 30;
    int button_spacing = 10;
    int start_x = 10;
    int start_y = 10;

    char red_label[] = "R";
    char green_label[] = "G";
    char blue_label[] = "B";
    char white_label[] = "W";
    char t_label[] = "T";
    char o_label[] = "O";
    char empty[] = "";


    Button red_button({start_x, start_y}, button_size, button_size, red_label, white, red, set_red_color);
    Button green_button({start_x + button_size + button_spacing, start_y}, button_size, button_size, green_label, white, green, set_green_color);
    Button blue_button({start_x + 2 * (button_size + button_spacing), start_y}, button_size, button_size, blue_label, white, blue, set_blue_color);
    Button white_button({start_x + 3 * (button_size + button_spacing), start_y}, button_size, button_size, white_label, white, black, set_white_color);
    Button T_button({start_x + 30, 260}, button_size, button_size, t_label, white, black, nullptr);
    Button O_button({420, 260}, button_size, button_size, o_label, white, black, nullptr);


    red_button.draw();
    green_button.draw();
    blue_button.draw();
    white_button.draw();
    T_button.draw();
    O_button.draw();


    Button selected_indicator({start_x, start_y + button_size + button_spacing + 10}, button_size, button_size, empty, selected_color, selected_color, nullptr);
    selected_indicator.draw();


    cts_points_t touch_points;
    Triangle* current_triangle = nullptr;
    Ellipse* current_ellipse = nullptr;

    while (1) {
        int num_points = cts_get_ts_points(&touch_points);
        if (num_points > 0 && touch_points.m_points[0].size > 0) {
            int touch_x = touch_points.m_points[0].x;
            int touch_y = touch_points.m_points[0].y;
            Point2D touch_point = {touch_x, touch_y};

            bool button_pressed = false;

            if (red_button.contains(touch_point)) {
                red_button.press();
                button_pressed = true;
                while (cts_get_ts_points(&touch_points) > 0 && touch_points.m_points[0].size > 0);
                red_button.release();
                selected_indicator.m_fg_color = selected_color;
                selected_indicator.m_bg_color = selected_color;
                selected_indicator.draw();
            } else if (green_button.contains(touch_point)) {
                green_button.press();
                button_pressed = true;
                while (cts_get_ts_points(&touch_points) > 0 && touch_points.m_points[0].size > 0);
                green_button.release();
                selected_indicator.m_fg_color = selected_color;
                selected_indicator.m_bg_color = selected_color;
                selected_indicator.draw();
            } else if (blue_button.contains(touch_point)) {
                blue_button.press();
                button_pressed = true;
                while (cts_get_ts_points(&touch_points) > 0 && touch_points.m_points[0].size > 0);
                blue_button.release();
                selected_indicator.m_fg_color = selected_color;
                selected_indicator.m_bg_color = selected_color;
                selected_indicator.draw();
            } else if (white_button.contains(touch_point)) {
                white_button.press();
                button_pressed = true;
                while (cts_get_ts_points(&touch_points) > 0 && touch_points.m_points[0].size > 0);
                white_button.release();
                selected_indicator.m_fg_color = selected_color;
                selected_indicator.m_bg_color = selected_color;
                selected_indicator.draw();
            } else if (T_button.contains(touch_point)) {
                T_button.press();
                button_pressed = true;
                while (cts_get_ts_points(&touch_points) > 0 && touch_points.m_points[0].size > 0);
                T_button.release();

                Point2D p1 = {LCD_WIDTH / 2 - 25, LCD_HEIGHT / 2 +30};
                Point2D p2 = {LCD_WIDTH / 2 + 25, LCD_HEIGHT / 2 +30 };
                Point2D p3 = {LCD_WIDTH / 2, LCD_HEIGHT / 2 + 100};
                current_triangle = new Triangle(p1, p2, p3, selected_color, black);
                current_triangle->draw();
            } else if (O_button.contains(touch_point)) {
                O_button.press();
                button_pressed = true;
                while (cts_get_ts_points(&touch_points) > 0 && touch_points.m_points[0].size > 0);
                O_button.release();

                current_ellipse = new Ellipse({LCD_WIDTH / 2, LCD_HEIGHT / 2}, 40, 25, selected_color, black);
                current_ellipse->draw();
            }


            if (current_triangle != nullptr && current_triangle->is_point_inside_triangle(touch_point)) {
                current_triangle->fill_shape(selected_color);
            } else if (current_ellipse != nullptr && current_ellipse->is_point_inside_ellipse(touch_point)) {
                current_ellipse->fill_shape(selected_color);
            }
        }
    }
}
