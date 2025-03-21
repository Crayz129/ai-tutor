"""
Скрипт для тестирования ИИ-тьютора по математике для подготовки к ЕГЭ.
"""

import os
import json
import sys
from src.agent.tutor_agent import MathTutorAgent
from src.agent.memory import MathTutorMemory
from src.utils.vector_store import MathProblemVectorStore
from dotenv import load_dotenv

load_dotenv()

def test_agent_initialization():
    """
    Тестирование инициализации агента.
    """
    print("Тестирование инициализации агента...")
    
    try:
        agent = MathTutorAgent()
        print("✓ Агент успешно инициализирован")
    except Exception as e:
        print(f"✗ Ошибка при инициализации агента: {str(e)}")
        return False
    
    return True

def test_memory_system():
    """
    Тестирование системы памяти.
    """
    print("\nТестирование системы памяти...")
    
    try:
        memory = MathTutorMemory()
        
        # Тестирование добавления задачи
        memory.add_problem(
            problem_id="test_1",
            problem_text="Решите уравнение: x^2 - 4 = 0",
            problem_type="алгебра",
            difficulty="Базовый уровень"
        )
        
        # Тестирование обновления прогресса
        memory.update_problem_progress(
            problem_id="test_1",
            progress_update="Я начал решать это уравнение, перенес 4 в правую часть: x^2 = 4",
            is_correct=True
        )
        
        # Тестирование добавления подсказки
        memory.add_hint(
            problem_id="test_1",
            hint="Подумайте, как найти квадратный корень из 4"
        )
        
        # Тестирование добавления ошибки
        memory.add_error(
            problem_id="test_1",
            error="Неправильно вычислен корень",
            concept="квадратный корень"
        )
        
        # Тестирование добавления освоенной концепции
        memory.add_mastered_concept("квадратное уравнение")
        
        # Тестирование получения контекста
        context = memory.get_context_for_problem("test_1")
        
        print("✓ Система памяти работает корректно")
    except Exception as e:
        print(f"✗ Ошибка при тестировании системы памяти: {str(e)}")
        return False
    
    return True

def test_vector_store():
    """
    Тестирование векторного хранилища.
    """
    print("\nТестирование векторного хранилища...")
    
    try:
        # Создание временной директории для тестирования
        test_dir = "./data/test_chroma_db"
        os.makedirs(test_dir, exist_ok=True)
        
        # Инициализация векторного хранилища
        vector_store = MathProblemVectorStore(persist_dir=test_dir)
        
        # Загрузка тестовых задач
        test_problems_file = "./data/test_problems.json"
        
        if os.path.exists(test_problems_file):
            count = vector_store.load_problems_from_json(test_problems_file)
            print(f"✓ Загружено {count} тестовых задач")
        else:
            print("✗ Файл с тестовыми задачами не найден")
            return False
        
        # Тестирование поиска задач
        results = vector_store.search_problems("квадратное уравнение", top_k=2)
        print(f"✓ Найдено {len(results)} задач по запросу 'квадратное уравнение'")
        
        # Тестирование поиска решений
        results = vector_store.search_solutions("объем пирамиды", top_k=2)
        print(f"✓ Найдено {len(results)} решений по запросу 'объем пирамиды'")
        
        print("✓ Векторное хранилище работает корректно")
    except Exception as e:
        print(f"✗ Ошибка при тестировании векторного хранилища: {str(e)}")
        return False
    
    return True

def test_chat_functionality():
    """
    Тестирование функциональности чата.
    """
    print("\nТестирование функциональности чата...")
    
    try:
        agent = MathTutorAgent()
        
        # Тестовые запросы
        test_queries = [
            "Что такое квадратное уравнение?",
            "Как решить уравнение x^2 - 4 = 0?",
            "Я получил ответ x = 2 и x = -2, правильно ли это?",
            "Можешь дать подсказку, как найти объем пирамиды?"
        ]
        
        for query in test_queries:
            print(f"\nЗапрос: {query}")
            response = agent.chat(query)
            print(f"Ответ: {response[:100]}...")  # Показываем только начало ответа
        
        print("✓ Функциональность чата работает корректно")
    except Exception as e:
        print(f"✗ Ошибка при тестировании функциональности чата: {str(e)}")
        return False
    
    return True

def run_all_tests():
    """
    Запуск всех тестов.
    """
    print("=== Тестирование ИИ-тьютора по математике для подготовки к ЕГЭ ===\n")
    
    tests = [
        test_agent_initialization,
        test_memory_system,
        test_vector_store,
        test_chat_functionality
    ]
    
    success_count = 0
    
    for test in tests:
        if test():
            success_count += 1
    
    print(f"\nРезультаты тестирования: {success_count}/{len(tests)} тестов пройдено успешно")
    
    if success_count == len(tests):
        print("\n✓ Все тесты пройдены успешно! ИИ-тьютор готов к использованию.")
        return True
    else:
        print("\n✗ Некоторые тесты не пройдены. Необходимо исправить ошибки.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
