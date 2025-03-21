"""
Логика тьютора для математических задач ЕГЭ.
"""

from typing import List, Dict, Any, Optional
import re
from llama_index.core.llms import LLM
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

class MathTutorLogic:
    """
    Логика тьютора для математических задач ЕГЭ.
    
    Эта логика определяет, как агент взаимодействует с учеником при решении задач,
    включая анализ задач, проверку решений, генерацию подсказок и определение
    следующих шагов.
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """
        Инициализация логики тьютора.
        
        Args:
            llm: Языковая модель для использования. По умолчанию используется OpenAI.
        """
        # Инициализация языковой модели
        self.llm = llm or OpenAI(model="gpt-4")
        Settings.llm = self.llm
        
        # Шаблоны для распознавания типов задач
        self.task_patterns = {
            "алгебра": [
                r"уравнени[еяй]",
                r"неравенств[оа]",
                r"функци[яий]",
                r"график",
                r"производн[аяой]",
                r"интеграл",
                r"логарифм",
                r"степен[ьи]"
            ],
            "геометрия": [
                r"треугольник",
                r"четырехугольник",
                r"окружность",
                r"многоугольник",
                r"призм[аы]",
                r"пирамид[аы]",
                r"конус",
                r"цилиндр",
                r"шар",
                r"объем",
                r"площадь",
                r"угол"
            ],
            "вероятность": [
                r"вероятност[ьи]",
                r"комбинаторик[аи]",
                r"сочетани[яей]",
                r"размещени[яей]",
                r"перестановк[аи]",
                r"событи[яей]",
                r"независим[ыоеая]"
            ],
            "текстовая_задача": [
                r"процент",
                r"кредит",
                r"вклад",
                r"скорост[ьи]",
                r"движени[еяй]",
                r"работ[аы]",
                r"производительност[ьи]"
            ]
        }
        
        # Шаблоны для распознавания ошибок
        self.error_patterns = {
            "алгебраические_ошибки": [
                r"знак[аи]",
                r"скобк[иа]",
                r"раскрыти[еяй]",
                r"приведени[еяй] подобных",
                r"сокращени[еяй]"
            ],
            "вычислительные_ошибки": [
                r"вычислени[еяй]",
                r"счет[аы]",
                r"арифметик[аи]"
            ],
            "логические_ошибки": [
                r"логик[аи]",
                r"следстви[еяй]",
                r"вывод[аы]",
                r"обоснован[ияей]"
            ],
            "концептуальные_ошибки": [
                r"понимани[еяй]",
                r"концепци[яий]",
                r"теор[ияем]",
                r"определени[еяй]"
            ]
        }
    
    def analyze_problem(self, problem_text: str) -> Dict[str, Any]:
        """
        Анализирует математическую задачу.
        
        Args:
            problem_text: Текст задачи.
            
        Returns:
            Dict[str, Any]: Информация о задаче.
        """
        # Определение типа задачи на основе шаблонов
        task_type = self._determine_task_type(problem_text)
        
        # Определение сложности задачи
        difficulty = self._determine_difficulty(problem_text, task_type)
        
        # Определение ключевых концепций
        key_concepts = self._determine_key_concepts(problem_text, task_type)
        
        # Формирование запроса к LLM для более глубокого анализа
        prompt = f"""
        Проанализируй следующую задачу ЕГЭ по математике:
        
        {problem_text}
        
        Предварительный анализ:
        - Тип задачи: {task_type}
        - Сложность: {difficulty}
        - Ключевые концепции: {', '.join(key_concepts)}
        
        Пожалуйста, предоставь более детальный анализ:
        1. Уточни тип задачи и подтип
        2. Оцени сложность по шкале от 1 до 5
        3. Определи все ключевые математические концепции, необходимые для решения
        4. Опиши возможные подходы к решению (без решения самой задачи)
        5. Укажи типичные ошибки, которые ученики могут допустить при решении
        
        Формат ответа: JSON
        """
        
        # Запрос к LLM
        response = self.llm.complete(prompt)
        
        # Попытка извлечь JSON из ответа
        try:
            import json
            detailed_analysis = json.loads(response.text)
        except:
            # Если не удалось извлечь JSON, используем предварительный анализ
            detailed_analysis = {
                "type": task_type,
                "subtype": "Требуется уточнение",
                "difficulty": difficulty,
                "key_concepts": key_concepts,
                "approaches": ["Требуется анализ"],
                "typical_errors": ["Требуется анализ"]
            }
        
        return detailed_analysis
    
    def check_solution(self, problem_text: str, solution_text: str, problem_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Проверяет решение математической задачи.
        
        Args:
            problem_text: Текст задачи.
            solution_text: Текст решения.
            problem_analysis: Анализ задачи (если уже выполнен).
            
        Returns:
            Dict[str, Any]: Результат проверки.
        """
        # Если анализ задачи не предоставлен, выполняем его
        if problem_analysis is None:
            problem_analysis = self.analyze_problem(problem_text)
        
        # Формирование запроса к LLM для проверки решения
        prompt = f"""
        Проверь решение следующей задачи ЕГЭ по математике:
        
        Задача:
        {problem_text}
        
        Решение ученика:
        {solution_text}
        
        Анализ задачи:
        - Тип задачи: {problem_analysis.get('type', 'Не определен')}
        - Сложность: {problem_analysis.get('difficulty', 'Не определена')}
        - Ключевые концепции: {', '.join(problem_analysis.get('key_concepts', ['Не определены']))}
        
        Пожалуйста, проверь решение и предоставь:
        1. Оценку правильности решения (правильно/частично правильно/неправильно)
        2. Список ошибок, если они есть
        3. Комментарии к каждому шагу решения
        4. Предложения по улучшению решения
        5. Оценку оформления решения с точки зрения требований ЕГЭ
        
        Важно: Не предоставляй полное правильное решение, только анализ и комментарии.
        
        Формат ответа: JSON
        """
        
        # Запрос к LLM
        response = self.llm.complete(prompt)
        
        # Попытка извлечь JSON из ответа
        try:
            import json
            check_result = json.loads(response.text)
        except:
            # Если не удалось извлечь JSON, используем базовый результат
            check_result = {
                "correctness": "Требуется проверка",
                "errors": [],
                "comments": ["Требуется анализ решения"],
                "suggestions": ["Требуется анализ решения"],
                "formatting": "Требуется оценка"
            }
        
        return check_result
    
    def generate_hint(self, problem_text: str, current_progress: str, problem_analysis: Optional[Dict[str, Any]] = None, student_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Генерирует подсказку для решения математической задачи.
        
        Args:
            problem_text: Текст задачи.
            current_progress: Текущий прогресс решения.
            problem_analysis: Анализ задачи (если уже выполнен).
            student_context: Контекст ученика (если доступен).
            
        Returns:
            Dict[str, Any]: Подсказка.
        """
        # Если анализ задачи не предоставлен, выполняем его
        if problem_analysis is None:
            problem_analysis = self.analyze_problem(problem_text)
        
        # Формирование контекста ученика для подсказки
        student_context_str = ""
        if student_context:
            difficult_concepts = student_context.get('difficult_concepts', [])
            mastered_concepts = student_context.get('mastered_concepts', [])
            
            if difficult_concepts:
                student_context_str += f"Концепции, с которыми ученик испытывает трудности: {', '.join(difficult_concepts)}\n"
            
            if mastered_concepts:
                student_context_str += f"Концепции, которые ученик уже освоил: {', '.join(mastered_concepts)}\n"
        
        # Формирование запроса к LLM для генерации подсказки
        prompt = f"""
        Сгенерируй подсказку для ученика, решающего следующую задачу ЕГЭ по математике:
        
        Задача:
        {problem_text}
        
        Текущий прогресс ученика:
        {current_progress}
        
        Анализ задачи:
        - Тип задачи: {problem_analysis.get('type', 'Не определен')}
        - Сложность: {problem_analysis.get('difficulty', 'Не определена')}
        - Ключевые концепции: {', '.join(problem_analysis.get('key_concepts', ['Не определены']))}
        
        {student_context_str}
        
        Пожалуйста, сгенерируй подсказку, которая:
        1. Не дает прямого ответа или полного решения
        2. Направляет ученика к следующему шагу
        3. Помогает преодолеть текущую трудность
        4. Учитывает уровень понимания ученика
        5. Содержит наводящий вопрос или мини-задачу
        
        Создай три уровня подсказок:
        - Легкая подсказка: минимальная помощь, просто направление мысли
        - Средняя подсказка: более конкретная помощь с указанием метода или формулы
        - Сильная подсказка: почти решение, но с пробелами для самостоятельного заполнения
        
        Формат ответа: JSON
        """
        
        # Запрос к LLM
        response = self.llm.complete(prompt)
        
        # Попытка извлечь JSON из ответа
        try:
            import json
            hint_result = json.loads(response.text)
        except:
            # Если не удалось извлечь JSON, используем базовую подсказку
            hint_result = {
                "light_hint": "Подумайте, какие формулы могут быть применимы к этой задаче.",
                "medium_hint": "Рассмотрите возможность применения следующего подхода...",
                "strong_hint": "Начните с ... и продолжите, используя ..."
            }
        
        return hint_result
    
    def generate_explanation(self, concept: str, difficulty_level: str = "medium", student_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Генерирует объяснение математической концепции.
        
        Args:
            concept: Математическая концепция.
            difficulty_level: Уровень сложности объяснения (easy, medium, hard).
            student_context: Контекст ученика (если доступен).
            
        Returns:
            str: Объяснение концепции.
        """
        # Формирование контекста ученика для объяснения
        student_context_str = ""
        if student_context:
            difficult_concepts = student_context.get('difficult_concepts', [])
            mastered_concepts = student_context.get('mastered_concepts', [])
            
            if difficult_concepts:
                student_context_str += f"Концепции, с которыми ученик испытывает трудности: {', '.join(difficult_concepts)}\n"
            
            if mastered_concepts:
                student_context_str += f"Концепции, которые ученик уже освоил: {', '.join(mastered_concepts)}\n"
        
        # Формирование запроса к LLM для генерации объяснения
        prompt = f"""
        Объясни следующую математическую концепцию для ученика, готовящегося к ЕГЭ по математике:
        
        Концепция: {concept}
        Уровень сложности объяснения: {difficulty_level}
        
        {student_context_str}
        
        Пожалуйста, предоставь объяснение, которое:
        1. Начинается с простого определения
        2. Включает ключевые формулы и свойства
        3. Приводит примеры применения в задачах ЕГЭ
        4. Указывает на типичные ошибки и как их избежать
        5. Содержит мнемонические правила или подсказки для запоминания
        
        Объяснение должно быть адаптировано к уровню ученика и содержать конкретные примеры.
        """
        
        # Запрос к LLM
        response = self.llm.complete(prompt)
        
        return response.text
    
    def generate_next_steps(self, problem_text: str, current_progress: str, check_result: Dict[str, Any]) -> List[str]:
        """
        Генерирует рекомендации по следующим шагам для ученика.
        
        Args:
            problem_text: Текст задачи.
            current_progress: Текущий прогресс решения.
            check_result: Результат проверки решения.
            
        Returns:
            List[str]: Список рекомендаций.
        """
        # Формирование запроса к LLM для генерации рекомендаций
        prompt = f"""
        Предложи следующие шаги для ученика, решающего задачу ЕГЭ по математике:
        
        Задача:
        {problem_text}
        
        Текущий прогресс:
        {current_progress}
        
        Результат проверки:
        - Правильность: {check_result.get('correctness', 'Не определена')}
        - Ошибки: {', '.join(check_result.get('errors', ['Не определены']))}
        
        Пожалуйста, предложи 3-5 конкретных шагов, которые ученик должен предпринять дальше.
        Шаги должны быть конкретными, выполнимыми и направленными на улучшение решения.
        Не предлагай полное решение, только направление действий.
        
        Формат: список шагов, каждый с кратким обоснованием.
        """
        
        # Запрос к LLM
        response = self.llm.complete(prompt)
        
        # Разбиение ответа на отдельные шаги
        steps = [step.strip() for step in response.text.split('\n') if step.strip()]
        
        return steps
    
    def _determine_task_type(self, problem_text: str) -> str:
        """
        Определяет тип задачи на основе шаблонов.
        
        Args:
            problem_text: Текст задачи.
            
        Returns:
            str: Тип задачи.
        """
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, problem_text, re.IGNORECASE):
                    return task_type
        
        return "Требуется анализ"
    
    def _determine_difficulty(self, problem_text: str, task_type: str) -> str:
        """
        Определяет сложность задачи.
        
        Args:
            problem_text: Текст задачи.
            task_type: Тип задачи.
            
        Returns:
            str: Сложность задачи.
        """
        # Простая эвристика для определения сложности
        word_count = len(problem_text.split())
        
        if word_count < 30:
            return "Базовый уровень"
        elif word_count < 60:
            return "Средний уровень"
        else:
            return "Повышенный уровень"
    
    def _determine_key_concepts(self, problem_text: str, task_type: str) -> List[str]:
        """
        Определяет ключевые концепции задачи.
        
        Args:
            problem_text: Текст задачи.
            task_type: Тип задачи.
            
        Returns:
            List[str]: Список ключевых концепций.
        """
        # Заглушка для будущей реализации
        # В реальной реализации здесь будет использоваться более сложная логика
        # для определения ключевых концепций на основе текста задачи и типа задачи
        return ["Требуется анализ"]
