# ИИ-тьютор по математике для подготовки к ЕГЭ

Этот проект представляет собой ИИ-агента на базе LlamaIndex, который выполняет роль тьютора по математике для подготовки к ЕГЭ. Агент учитывает специфику экзамена и не дает прямых ответов на задачи, а направляет пользователя к правильному решению.

## Особенности

- **Учет специфики ЕГЭ**: Агент знает структуру экзамена, типы заданий и критерии оценивания
- **Педагогический подход**: Не дает прямых ответов, а помогает пользователю самостоятельно найти решение
- **Анализ решений**: Проверяет решения пользователя, находит ошибки и помогает их исправить
- **Персонализированные подсказки**: Генерирует подсказки разного уровня в зависимости от прогресса пользователя
- **Векторное хранилище**: Использует Chroma для хранения базы заданий и их решений
- **Удобный интерфейс**: Взаимодействие с агентом происходит через чат-интерфейс на базе Gradio

## Структура проекта

├── src/
│   ├── agent/
│   │   ├── init.py
│   │   ├── tutor_agent.py      # Основной класс агента-тьютора
│   │   ├── memory.py           # Система контекстной памяти
│   │   └── logic.py            # Логика тьютора для математических задач
│   ├── utils/
│   │   ├── init.py
│   │   └── vector_store.py     # Векторное хранилище для базы заданий
│   ├── app.py                  # Gradio интерфейс
│   └── init.py
├── data/
│   ├── problems.json           # База задач ЕГЭ
│   ├── concepts.json           # База математических концепций
│   └── chroma_db/              # Директория для векторной базы данных
├── main.py                     # Основной файл для запуска приложения
├── test_agent.py               # Скрипт для тестирования агента
└── README.md                   # Документация