"""
Основной файл для запуска ИИ-тьютора по математике для подготовки к ЕГЭ.
"""

import os
import argparse
from src.app import MathTutorApp

def main():
    """
    Основная функция для запуска приложения.
    """
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="ИИ-тьютор по математике для подготовки к ЕГЭ")
    parser.add_argument("--share", action="store_true", help="Создать публичную ссылку")
    parser.add_argument("--debug", action="store_true", help="Включить режим отладки")
    args = parser.parse_args()
    
    # Создание директорий, если они не существуют
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/chroma_db", exist_ok=True)
    
    # Настройка режима отладки
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Создание и запуск приложения
    app = MathTutorApp()
    app.launch(share=args.share)

if __name__ == "__main__":
    main()
