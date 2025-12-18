///
/// \author	John Farrier
///
/// \copyright Copyright 2015-2023 John Farrier
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///
/// http://www.apache.org/licenses/LICENSE-2.0
///
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.
///
#include "../include/Console.h"

#include <iostream>

void Red() {
    std::cout << "\033[49m\033[31m";
}

void RedBold() {
    std::cout << "\033[49m\033[1;31m";
}

void Green() {
    std::cout << "\033[49m\033[32m";
}

void GreenBold() {
    std::cout << "\033[49m\033[1;32m";
}

void Blue() {
    std::cout << "\033[49m\033[34m";
}

void BlueBold() {
    std::cout << "\033[49m\033[1;34m";
}

void Cyan() {
    std::cout << "\033[49m\033[36m";
}

void CyanBold() {
    std::cout << "\033[49m\033[1;36m";
}

void Yellow() {
    std::cout << "\033[49m\033[33m";
}

void YellowBold() {
    std::cout << "\033[49m\033[1;33m";
}

void White() {
    std::cout << "\033[49m\033[37m";
}

void WhiteBold() {
    std::cout << "\033[49m\033[1;37m";
}

void WhiteOnRed() {
    std::cout << "\033[41m\033[37m";
}

void WhiteOnRedBold() {
    std::cout << "\033[41m\033[1;37m";
}

void PurpleBold() {
    std::cout << "\033[49m\033[1;38m";
}

void Default() {
    std::cout << "\033[0m";
}

void console::SetConsoleColor(const console::ConsoleColor x) {
    switch (x) {
    case console::ConsoleColor::Red:
        Red();
        break;
    case console::ConsoleColor::Red_Bold:
        RedBold();
        break;
    case console::ConsoleColor::Green:
        Green();
        break;
    case console::ConsoleColor::Green_Bold:
        GreenBold();
        break;
    case console::ConsoleColor::Blue:
        Blue();
        break;
    case console::ConsoleColor::Blue_Bold:
        BlueBold();
        break;
    case console::ConsoleColor::Cyan:
        Cyan();
        break;
    case console::ConsoleColor::Cyan_Bold:
        CyanBold();
        break;
    case console::ConsoleColor::Yellow:
        Yellow();
        break;
    case console::ConsoleColor::Yellow_Bold:
        YellowBold();
        break;
    case console::ConsoleColor::White:
        White();
        break;
    case console::ConsoleColor::White_Bold:
        WhiteBold();
        break;
    case console::ConsoleColor::WhiteOnRed:
        WhiteOnRed();
        break;
    case console::ConsoleColor::WhiteOnRed_Bold:
        WhiteOnRedBold();
        break;
    case console::ConsoleColor::Purple_Bold:
        PurpleBold();
        break;
    case console::ConsoleColor::Default:
    default:
        Default();
        break;
    }
}
