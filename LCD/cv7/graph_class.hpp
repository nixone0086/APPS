#include "lcd_lib.h"
#include <cstdlib>
#include "font22x36_lsb.h"
struct Point2D {
    int32_t x, y;
};
struct RGB {
    uint8_t r, g, b;
};
class GraphElement {
public:
    RGB m_fg_color, m_bg_color;
    GraphElement(RGB t_fg_color, RGB t_bg_color)
        : m_fg_color(t_fg_color), m_bg_color(t_bg_color) {}
    virtual void draw() = 0;
    virtual void hide() {
        swap_fg_bg_color();
        draw();
        swap_fg_bg_color();
    }
private:
    void swap_fg_bg_color() {
        RGB tmp = m_fg_color;
        m_fg_color = m_bg_color;
        m_bg_color = tmp;
    }
};
class Pixel : public GraphElement {
public:
    Point2D m_pos;
    Pixel(Point2D t_pos, RGB t_fg_color, RGB t_bg_color)
        : GraphElement(t_fg_color, t_bg_color), m_pos(t_pos) {}
    void draw() override {
        uint16_t color = ((m_fg_color.r >> 3) << 11) |
                         ((m_fg_color.g >> 2) << 5) |
                         (m_fg_color.b >> 3);
        lcd_put_pixel(m_pos.x, m_pos.y, color);
    }
};
class Circle : public GraphElement {
public:
    Point2D m_center;
    int32_t m_radius;
    Circle(Point2D t_center, int32_t t_radius, RGB t_fg, RGB t_bg)
        : GraphElement(t_fg, t_bg), m_center(t_center), m_radius(t_radius) {}
    void draw() override {
        int x = m_radius;
        int y = 0;
        int err = 0;
        while (x >= y) {
            uint16_t color = ((m_fg_color.r >> 3) << 11) |
                             ((m_fg_color.g >> 2) << 5) |
                             (m_fg_color.b >> 3);
            lcd_put_pixel(m_center.x + x, m_center.y + y, color);
            lcd_put_pixel(m_center.x - x, m_center.y + y, color);
            lcd_put_pixel(m_center.x + x, m_center.y - y, color);
            lcd_put_pixel(m_center.x - x, m_center.y - y, color);
            lcd_put_pixel(m_center.x + y, m_center.y + x, color);
            lcd_put_pixel(m_center.x - y, m_center.y + x, color);
            lcd_put_pixel(m_center.x + y, m_center.y - x, color);
            lcd_put_pixel(m_center.x - y, m_center.y - x, color);
            y++;
            if (err <= 0) {
                err += 2 * y + 1;
            } else {
                x--;
                err += 2 * (y - x + 1);
            }
        }
    }
};
class Line : public GraphElement {
public:
    Point2D m_pos1, m_pos2;
    Line(Point2D t_pos1, Point2D t_pos2, RGB t_fg, RGB t_bg)
        : GraphElement(t_fg, t_bg), m_pos1(t_pos1), m_pos2(t_pos2) {}
    void draw() override {
        int dx = abs(m_pos2.x - m_pos1.x);
        int dy = -abs(m_pos2.y - m_pos1.y);
        int err = dx + dy;
        int e2;
        int x = m_pos1.x;
        int y = m_pos1.y;
        while (true) {
            uint16_t color = ((m_fg_color.r >> 3) << 11) |
                             ((m_fg_color.g >> 2) << 5) |
                             (m_fg_color.b >> 3);
            lcd_put_pixel(x, y, color);
            if (x == m_pos2.x && y == m_pos2.y) break;
            e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x += (m_pos1.x < m_pos2.x) ? 1 : -1;
            }
            if (e2 <= dx) {
                err += dx;
                y += (m_pos1.y < m_pos2.y) ? 1 : -1;
            }
        }
    }
};
class Char22x36 : public GraphElement {
public:
    Point2D m_pos;
    char m_char;
    Char22x36(Point2D t_pos, char t_char, RGB t_fg, RGB t_bg)
        : GraphElement(t_fg, t_bg), m_pos(t_pos), m_char(t_char) {}
    void draw() override {
        const uint32_t* ch_bitmap = font[(uint8_t)m_char];
        for (int row = 0; row < 36; ++row) {
            uint32_t row_data = ch_bitmap[row];
            for (int col = 0; col < 22; ++col) {
                if (row_data & (1 << col)) {
                    uint16_t rgb565 = ((m_fg_color.r >> 3) << 11) |
                                      ((m_fg_color.g >> 2) << 5) |
                                      (m_fg_color.b >> 3);
                    lcd_put_pixel(m_pos.x + col, m_pos.y + row, rgb565);
                }
            }
        }
     }

};
            class CSS : public GraphElement {
            public:
                Char22x36* m_chars[4];
                CSS(Point2D t_origin)
                    : GraphElement({0, 0, 0}, {255, 255, 255}) {
                    const int char_width = 22;
                    const int char_height = 36;
                    const int spacing = 4;
                    int start_x = LCD_WIDTH - char_width - 2;
                    int start_y = 10;
                    const char chars[] = {'R', 'G', 'B', 'W'};
                    const RGB colors[] = {
                        {255, 0, 0},
                        {0, 255, 0},
                        {0, 0, 255},
                        {255, 255, 255}
                    };
                    for (int i = 0; i < 4; ++i) {
                        Point2D pos = {start_x, start_y + i * (char_height + spacing)};
                        m_chars[i] = new Char22x36(pos, chars[i], colors[i], m_bg_color);
                    }
                }
                void draw() override {
                    for (int i = 0; i < 4; ++i) {
                        m_chars[i]->draw();
                    }
                }
                ~CSS() {
                    for (int i = 0; i < 4; ++i) {
                        delete m_chars[i];
                    }
                }
            };

            class Button : public GraphElement {
            public:
                Point2D m_top_left;
                int32_t m_width;
                int32_t m_height;
                char* m_label;
                bool m_pressed;
                void (*m_callback)();
                Button(Point2D t_top_left, int32_t t_width, int32_t t_height,
                       char* t_label, RGB t_fg, RGB t_bg, void (*t_callback)() = nullptr)
                    : GraphElement(t_fg, t_bg),
                      m_top_left(t_top_left),
                      m_width(t_width),
                      m_height(t_height),
                      m_label(t_label),
                      m_pressed(false),
                      m_callback(t_callback) {}
                void draw() override {

                    for (int y = 0; y < m_height; ++y) {
                        for (int x = 0; x < m_width; ++x) {
                            uint16_t color = ((m_fg_color.r >> 3) << 11) |
                                           ((m_fg_color.g >> 2) << 5) |
                                           (m_fg_color.b >> 3);

                            if (x < 2 || x >= m_width - 2 || y < 2 || y >= m_height - 2) {
                                lcd_put_pixel(m_top_left.x + x, m_top_left.y + y, color);
                            } else {

                                uint16_t bg_color = ((m_bg_color.r >> 3) << 11) |
                                                 ((m_bg_color.g >> 2) << 5) |
                                                 (m_bg_color.b >> 3);
                                lcd_put_pixel(m_top_left.x + x, m_top_left.y + y, bg_color);
                            }
                        }
                    }

                    if (m_label != nullptr) {
                        int label_length = strlen(m_label);
                        int char_width = 22;
                        int char_height = 36;

                        int start_x = m_top_left.x + (m_width - label_length * char_width) / 2;
                        int start_y = m_top_left.y + (m_height - char_height) / 2 ;

                        if (start_x < m_top_left.x) start_x = m_top_left.x + 4;
                        if (start_y < m_top_left.y) start_y = m_top_left.y ;
                        for (int i = 0; i < label_length && i * char_width < m_width - 8; ++i) {
                            Point2D char_pos = {start_x + i * char_width, start_y};
                            Char22x36 ch(char_pos, m_label[i], m_fg_color, m_bg_color);
                            ch.draw();
                        }
                    }
                }
                // Проверка, находится ли точка внутри кнопки
                bool contains(Point2D point) {
                    return (point.x >= m_top_left.x &&
                            point.x < m_top_left.x + m_width &&
                            point.y >= m_top_left.y &&
                            point.y < m_top_left.y + m_height);
                }
                // Обработка нажатия
                void press() {
                    m_pressed = true;
                    draw(); // Перерисовываем кнопку в нажатом состоянии
                }
                // Обработка отпускания
                void release() {
                    m_pressed = false;
                    draw(); // Возвращаем кнопку в исходное состояние
                    // Вызываем функцию обратного вызова, если она задана
                    if (m_callback != nullptr) {
                        m_callback();
                    }
                }
                // Метод для изменения функции обратного вызова
                void set_callback(void (*t_callback)()) {
                    m_callback = t_callback;
                }
            };
