#include <stdio.h>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <ctime>
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
#include <cstdio>



void _mcu_initialization() __attribute__((constructor(0x100)));

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
DigitalIn  g_but_P3_18(P3_18);
DigitalIn  g_but_P3_19(P3_19);
DigitalIn  g_but_P3_20(P3_20);
DigitalIn  g_but_P3_21(P3_21);

RGB red     = {255, 0, 0};
RGB green   = {0, 255, 0};
RGB blue    = {0, 0, 255};
RGB white   = {255, 255, 255};
RGB black   = {0, 0, 0};
RGB selected_color = red;

void set_red_color()   { selected_color = red; }
void set_green_color() { selected_color = green; }
void set_blue_color()  { selected_color = blue; }
void set_white_color() { selected_color = white; }


class Triangle : public GraphElement {
public:
    Point2D m_p1, m_p2, m_p3;

    Triangle(Point2D t_p1, Point2D t_p2, Point2D t_p3, RGB t_fg, RGB t_bg)
        : GraphElement(t_fg, t_bg), m_p1(t_p1), m_p2(t_p2), m_p3(t_p3) {}

    // Рисуем заполненный треугольник
    void draw() override {
        fill_shape(m_fg_color);
    }

    void fill_shape(RGB fill_color) {
        int minX = std::min({m_p1.x, m_p2.x, m_p3.x});
        int maxX = std::max({m_p1.x, m_p2.x, m_p3.x});
        int minY = std::min({m_p1.y, m_p2.y, m_p3.y});
        int maxY = std::max({m_p1.y, m_p2.y, m_p3.y});

        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                if (is_point_inside_triangle({x, y})) {
                    lcd_put_pixel(
                        x, y,
                        ((fill_color.r >> 3) << 11)
                        | ((fill_color.g >> 2) << 5)
                        | (fill_color.b >> 3)
                    );
                }
            }
        }
    }

    bool is_point_inside_triangle(Point2D p) {
        float area = fabs((m_p1.x * (m_p2.y - m_p3.y) +
                           m_p2.x * (m_p3.y - m_p1.y) +
                           m_p3.x * (m_p1.y - m_p2.y)) / 2.0);

        float area1 = fabs((p.x * (m_p2.y - m_p3.y) +
                            m_p2.x * (m_p3.y - p.y) +
                            m_p3.x * (p.y - m_p2.y)) / 2.0);

        float area2 = fabs((m_p1.x * (p.y - m_p3.y) +
                            p.x * (m_p3.y - m_p1.y) +
                            m_p3.x * (m_p1.y - p.y)) / 2.0);

        float area3 = fabs((m_p1.x * (m_p2.y - p.y) +
                            m_p2.x * (p.y - m_p1.y) +
                            p.x * (m_p1.y - m_p2.y)) / 2.0);

        return (fabs(area - (area1 + area2 + area3)) < 0.1);
    }
};



class Ellipse : public GraphElement {
public:
    Point2D m_center;
    int32_t m_a, m_b;

    Ellipse(Point2D t_center, int32_t t_a, int32_t t_b, RGB t_fg, RGB t_bg)
        : GraphElement(t_fg, t_bg), m_center(t_center), m_a(t_a), m_b(t_b) {}

    // Рисуем заполненный эллипс
    void draw() override {
        fill_shape(m_fg_color);
    }

    void fill_shape(RGB fill_color) {
        int minX = m_center.x - m_a;
        int maxX = m_center.x + m_a;
        int minY = m_center.y - m_b;
        int maxY = m_center.y + m_b;

        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                if (is_point_inside_ellipse({x, y})) {
                    lcd_put_pixel(
                        x, y,
                        ((fill_color.r >> 3) << 11)
                        | ((fill_color.g >> 2) << 5)
                        | (fill_color.b >> 3)
                    );
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



class Cannon : public GraphElement {
public:
    Point2D m_base;
    int m_length;
    float m_angle; // в градусах

    Cannon(Point2D t_base, int t_length, RGB t_fg, RGB t_bg)
        : GraphElement(t_fg, t_bg), m_base(t_base), m_length(t_length), m_angle(0) {}

    void draw() override {

        Circle base(m_base, 10, m_fg_color, m_fg_color);
        base.draw();

        float rad = m_angle * 3.14159f / 180.0f;
        int endX = m_base.x + sinf(rad) * m_length;
        int endY = m_base.y - cosf(rad) * m_length;

        // Вычисляем вектор ствола
        int dx = endX - m_base.x;
        int dy = endY - m_base.y;
        float len = sqrtf(dx*dx + dy*dy);
        if(len == 0) len = 1;
        // Нормализованный перпендикулярный вектор для утолщения ствола
        float offsetX = -dy / len;
        float offsetY = dx / len;
        int thickness = 2;

        // Рисуем несколько параллельных линий, смещённых по перпендикуляру
        for (int i = -thickness; i <= thickness; i++) {
            int ox = round(offsetX * i);
            int oy = round(offsetY * i);
            Point2D newBase = { m_base.x + ox, m_base.y + oy };
            Point2D newEnd  = { endX + ox, endY + oy };
            Line barrel(newBase, newEnd, m_fg_color, m_bg_color);
            barrel.draw();
        }
    }

    void setAngle(float angle) {
        hide();
        m_angle = (angle < -60) ? -60 : ((angle > 60) ? 60 : angle);
        draw();
    }

    Point2D getBarrelEnd() {
        float rad = m_angle * 3.14159f / 180.0f;
        int endX = m_base.x + sinf(rad) * m_length;
        int endY = m_base.y - cosf(rad) * m_length;
        return {endX, endY};
    }
};

class Scrollbar : public GraphElement {
public:
    Point2D m_pos;
    int m_width;
    int m_height;
    int m_value; // 0..100

    Scrollbar(Point2D t_pos, int t_width, int t_height, RGB t_fg, RGB t_bg)
        : GraphElement(t_fg, t_bg), m_pos(t_pos), m_width(t_width), m_height(t_height), m_value(50) {}

    void draw() override {
        // Трек (небольшой прямоугольник высотой 4px)
        int trackHeight = 4;
        int trackY = m_pos.y + m_height / 2 - trackHeight / 2;
        for (int y = trackY; y < trackY + trackHeight; y++) {
            for (int x = m_pos.x; x < m_pos.x + m_width; x++) {
                lcd_put_pixel(
                    x, y,
                    ((m_fg_color.r >> 3) << 11)
                    | ((m_fg_color.g >> 2) << 5)
                    | (m_fg_color.b >> 3)
                );
            }
        }
        // Рисуем handle (ползунок) как круг
        int handlePos = m_pos.x + (m_width * m_value) / 100;
        Circle handle({handlePos, m_pos.y + m_height / 2}, m_height / 2, m_fg_color, m_bg_color);
        handle.draw();
    }

    bool processTouch(Point2D touchPoint) {
        if (touchPoint.y >= m_pos.y && touchPoint.y <= m_pos.y + m_height &&
            touchPoint.x >= m_pos.x && touchPoint.x <= m_pos.x + m_width) {
            hide();
            m_value = ((touchPoint.x - m_pos.x) * 100) / m_width;
            if (m_value < 0)   m_value = 0;
            if (m_value > 100) m_value = 100;
            draw();
            return true;
        }
        return false;
    }

    int getValue() {
        return m_value;
    }

    float getAngle() {
        // Преобразуем 0..100 в -60..+60
        return -60 + (m_value * 120) / 100;
    }
};


class Projectile : public GraphElement {
public:
    Point2D m_pos;
    float m_angle;
    float m_speed;
    int m_radius;
    bool m_active;

    Projectile(Point2D t_pos, float t_angle, float t_speed, int t_radius, RGB t_color, RGB t_bg)
        : GraphElement(t_color, t_bg), m_pos(t_pos), m_angle(t_angle),
          m_speed(t_speed), m_radius(t_radius), m_active(true) {}

    virtual ~Projectile() {}

    void draw() override {
        Circle projectile(m_pos, m_radius, m_fg_color, m_bg_color);
        projectile.draw();
    }

    bool update() {
        if (!m_active) return false;
        hide();
        float rad = m_angle * 3.14159f / 180.0f;
        m_pos.x += m_speed * sinf(rad);
        m_pos.y -= m_speed * cosf(rad);
        if (m_pos.x < 0 || m_pos.x > LCD_WIDTH || m_pos.y < 0 || m_pos.y > LCD_HEIGHT) {
            m_active = false;
            return false;
        }
        draw();
        return true;
    }

    bool isActive() {
        return m_active;
    }

    void deactivate() {
        hide();
        m_active = false;
    }
};


class Game {
private:
    Cannon* m_cannon;
    Scrollbar* m_scrollbar;
    Button* m_fireButton;
    Projectile* m_projectile;
    GraphElement* m_target;
    bool m_isTargetTriangle;
    RGB m_selectedColor;
    int m_score;
    bool m_gameRunning;
    char m_scoreText[20];

public:
    Game() {

        m_cannon = new Cannon({LCD_WIDTH / 2, LCD_HEIGHT - 80}, 30, white, black);

        m_scrollbar = new Scrollbar({LCD_WIDTH / 2 - 100, LCD_HEIGHT - 50}, 200, 15, white, black);

        m_fireButton = new Button({(LCD_WIDTH / 2) - 55, LCD_HEIGHT - 30}, 110, 30, "FIRE!", white, red, nullptr);

        m_projectile = nullptr;
        m_target = nullptr;
        m_selectedColor = red;
        m_score = 0;
        m_gameRunning = true;
        sprintf(m_scoreText, "Score: %d", m_score);
    }

    ~Game() {
        if (m_cannon)     delete m_cannon;
        if (m_scrollbar)  delete m_scrollbar;
        if (m_fireButton) delete m_fireButton;
        if (m_projectile) delete m_projectile;
        if (m_target)     delete m_target;
    }

    void init() {
        m_cannon->draw();
        m_scrollbar->draw();
        m_fireButton->draw();
        generateTarget();
    }

    void generateTarget() {
        if (m_target) {
            m_target->hide();
            delete m_target;
            m_target = nullptr;
        }
        int x1 = 50 + rand() % (LCD_WIDTH - 100);
        int y1 = 50 + rand() % (LCD_HEIGHT / 2 - 100);
        RGB colors[] = {red, green, blue, white};
        RGB targetColor = colors[rand() % 4];
        bool isTriangle = (rand() % 2 == 0);
        m_isTargetTriangle = isTriangle;

        if (isTriangle) {
            Point2D p1 = {x1, y1};
            Point2D p2 = {x1 + 30, y1 + 20};
            Point2D p3 = {x1 - 10, y1 + 40};
            m_target = new Triangle(p1, p2, p3, targetColor, black);
        } else {
            Point2D center = {x1, y1};
            m_target = new Ellipse(center, 40, 25, targetColor, black);
        }
        m_target->draw();
    }

    void setSelectedColor(RGB color) {
        m_selectedColor = color;
    }

    void processTouch(Point2D touchPoint) {
        if (m_scrollbar->processTouch(touchPoint)) {
            // меняем угол пушки
            m_cannon->setAngle(m_scrollbar->getAngle());
        } else if (m_fireButton->contains(touchPoint)) {
            m_fireButton->press();
            fire();
            m_fireButton->release();
        }
    }

    void fire() {
        if (m_projectile && m_projectile->isActive()) return;
        if (m_projectile) {
            delete m_projectile;
            m_projectile = nullptr;
        }
        Point2D barrelEnd = m_cannon->getBarrelEnd();
        m_projectile = new Projectile(barrelEnd, m_cannon->m_angle, 5.0f, 5, m_selectedColor, black);
        m_projectile->draw();
    }

    void update() {
        if (!m_gameRunning) return;
        if (m_projectile && m_projectile->isActive()) {
            m_projectile->update();
            if (m_target && checkCollision()) {
                handleCollision();
            }
        }
    }

    bool checkCollision() {
        if (!m_projectile || !m_projectile->isActive() || !m_target) return false;
        if (m_isTargetTriangle) {
            Triangle* tri = static_cast<Triangle*>(m_target);
            return tri->is_point_inside_triangle(m_projectile->m_pos);
        } else {
            Ellipse* ellipse = static_cast<Ellipse*>(m_target);
            return ellipse->is_point_inside_ellipse(m_projectile->m_pos);
        }
    }

    void handleCollision() {
        m_projectile->deactivate();
        RGB targetColor;
        if (m_isTargetTriangle) {
            auto tri = static_cast<Triangle*>(m_target);
            targetColor = tri->m_fg_color;
        } else {
            auto ellipse = static_cast<Ellipse*>(m_target);
            targetColor = ellipse->m_fg_color;
        }
        // Бонус, если цвет совпал
        if (targetColor.r == m_projectile->m_fg_color.r &&
            targetColor.g == m_projectile->m_fg_color.g &&
            targetColor.b == m_projectile->m_fg_color.b) {
            m_score += 10;
        } else {
            m_score += 5;
        }
        updateScore();
        generateTarget();
    }

    void updateScore() {
        sprintf(m_scoreText, "Score: %d", m_score);

        for (int y = 10; y < 30; y++) {
            for (int x = LCD_WIDTH - 100; x < LCD_WIDTH - 10; x++) {
                lcd_put_pixel(
                    x, y,
                    ((black.r >> 3) << 11)
                    | ((black.g >> 2) << 5)
                    | (black.b >> 3)
                );
            }
        }

        int startX = 270;
        int letterSpacing = 20;
        for (int i = 0; m_scoreText[i] != '\0'; i++) {
            Char22x36 ch({startX + i * letterSpacing, 10}, m_scoreText[i], white, black);
            ch.draw();
        }
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
    srand(time(NULL));

    Game game;
    game.init();

    // Кнопки выбора цвета
    int button_size = 30;
    int button_spacing = 10;
    int start_x = 10;
    int start_y = 10;

    char red_label[]   = "R";
    char green_label[] = "G";
    char blue_label[]  = "B";
    char white_label[] = "W";
    char empty[]       = "";

    Button red_button  ({start_x, start_y},
                        button_size, button_size,
                        red_label, white, red,
                        set_red_color);

    Button green_button({start_x + button_size + button_spacing, start_y},
                        button_size, button_size,
                        green_label, white, green,
                        set_green_color);

    Button blue_button ({start_x + 2*(button_size + button_spacing), start_y},
                        button_size, button_size,
                        blue_label, white, blue,
                        set_blue_color);

    Button white_button({start_x + 3*(button_size + button_spacing), start_y},
                        button_size, button_size,
                        white_label, white, black,
                        set_white_color);

    red_button.draw();
    green_button.draw();
    blue_button.draw();
    white_button.draw();

    // Индикатор текущего цвета
    Button selected_indicator({start_x + 160, start_y},
                              button_size, button_size,
                              empty, selected_color, selected_color,
                              nullptr);
    selected_indicator.draw();

    cts_points_t touch_points;

    while (1) {
        int num_points = cts_get_ts_points(&touch_points);
        if (num_points > 0 && touch_points.m_points[0].size > 0) {
            int touch_x = touch_points.m_points[0].x;
            int touch_y = touch_points.m_points[0].y;
            Point2D touch_point = {touch_x, touch_y};
            bool button_pressed = false;

            // Проверяем, не нажата ли одна из кнопок выбора цвета
            if (red_button.contains(touch_point)) {
                red_button.press();
                button_pressed = true;
                while (cts_get_ts_points(&touch_points) > 0 && touch_points.m_points[0].size > 0);
                red_button.release();
                selected_color = red;
                game.setSelectedColor(red);
                selected_indicator.m_fg_color = selected_color;
                selected_indicator.m_bg_color = selected_color;
                selected_indicator.draw();
            }
            else if (green_button.contains(touch_point)) {
                green_button.press();
                button_pressed = true;
                while (cts_get_ts_points(&touch_points) > 0 && touch_points.m_points[0].size > 0);
                green_button.release();
                selected_color = green;
                game.setSelectedColor(green);
                selected_indicator.m_fg_color = selected_color;
                selected_indicator.m_bg_color = selected_color;
                selected_indicator.draw();
            }
            else if (blue_button.contains(touch_point)) {
                blue_button.press();
                button_pressed = true;
                while (cts_get_ts_points(&touch_points) > 0 && touch_points.m_points[0].size > 0);
                blue_button.release();
                selected_color = blue;
                game.setSelectedColor(blue);
                selected_indicator.m_fg_color = selected_color;
                selected_indicator.m_bg_color = selected_color;
                selected_indicator.draw();
            }
            else if (white_button.contains(touch_point)) {
                white_button.press();
                button_pressed = true;
                while (cts_get_ts_points(&touch_points) > 0 && touch_points.m_points[0].size > 0);
                white_button.release();
                selected_color = white;
                game.setSelectedColor(white);
                selected_indicator.m_fg_color = selected_color;
                selected_indicator.m_bg_color = selected_color;
                selected_indicator.draw();
            }

            if (!button_pressed) {
                game.processTouch(touch_point);
            }
        }


        game.update();

        if (!g_but_P3_18.read()) {

            game.fire();
            delay_ms(200);
        }

        if (!g_but_P3_19.read()) {
            if (selected_color.r == red.r && selected_color.g == red.g && selected_color.b == red.b) {
                selected_color = green;
            } else if (selected_color.r == green.r && selected_color.g == green.g && selected_color.b == green.b) {
                selected_color = blue;
            } else if (selected_color.r == blue.r && selected_color.g == blue.g && selected_color.b == blue.b) {
                selected_color = white;
            } else {
                selected_color = red;
            }
            game.setSelectedColor(selected_color);
            selected_indicator.m_fg_color = selected_color;
            selected_indicator.m_bg_color = selected_color;
            selected_indicator.draw();

            g_led_P3_16.write(selected_color.r > 0 ? 1 : 0);
            g_led_P3_17.write(selected_color.g > 0 ? 1 : 0);

            delay_ms(200);
        }

        if (!g_but_P3_20.read()) {
            for (int y = 0; y < LCD_HEIGHT; y++) {
                for (int x = 0; x < LCD_WIDTH; x++) {
                    lcd_put_pixel(
                        x, y,
                        ((black.r >> 3) << 11)
                        | ((black.g >> 2) << 5)
                        | (black.b >> 3)
                    );
                }
            }
            game = Game();
            game.init();
            red_button.draw();
            green_button.draw();
            blue_button.draw();
            white_button.draw();
            selected_indicator.draw();
            delay_ms(200);
        }

        if (!g_but_P3_21.read()) {
            static bool paused = false;
            paused = !paused;
            if (paused) {
                char paused_text[] = "PAUSED";
                for (int i = 0; paused_text[i] != '\0'; i++) {
                    Char22x36 ch({LCD_WIDTH/2 - 70 + i*20, LCD_HEIGHT/2 - 18}, paused_text[i], white, black);
                    ch.draw();
                }

                while (g_but_P3_21.read());
                while (!g_but_P3_21.read());

                for (int y = LCD_HEIGHT/2 - 20; y < LCD_HEIGHT/2 + 20; y++) {
                    for (int x = LCD_WIDTH/2 - 80; x < LCD_WIDTH/2 + 80; x++) {
                        lcd_put_pixel(
                            x, y,
                            ((black.r >> 3) << 11)
                            | ((black.g >> 2) << 5)
                            | (black.b >> 3)
                        );
                    }
                }
            }
            delay_ms(200);
        }

        delay_ms(10);
    }
    return 0;
}
