"""
Gradio интерфейс для ИИ-тьютора по математике для подготовки к ЕГЭ.
"""

import os
import gradio as gr
import tempfile
from PIL import Image
import pytesseract
from datetime import datetime

# Импорт компонентов агента
from src.agent.tutor_agent import MathTutorAgent
from src.utils.vector_store import MathProblemVectorStore

class MathTutorApp:
    """
    Приложение для ИИ-тьютора по математике для подготовки к ЕГЭ.
    """
    
    def __init__(self):
        """
        Инициализация приложения.
        """
        # Инициализация агента
        self.agent = MathTutorAgent()
        
        # Инициализация векторного хранилища
        self.vector_store = MathProblemVectorStore()
        
        # Загрузка базы задач (если есть)
        problems_json = os.path.join("data", "problems.json")
        if os.path.exists(problems_json):
            self.vector_store.load_problems_from_json(problems_json)
        
        # Загрузка базы концепций (если есть)
        concepts_json = os.path.join("data", "concepts.json")
        if os.path.exists(concepts_json):
            self.vector_store.load_concepts_from_json(concepts_json)
        
        # Создание интерфейса
        self.create_interface()
    
    def create_interface(self):
        """
        Создание Gradio интерфейса.
        """
        # Определение CSS стилей
        css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .math-tutor-header {
            text-align: center;
            margin-bottom: 1rem;
        }
        .math-tutor-description {
            margin-bottom: 1.5rem;
            text-align: justify;
        }
        .math-tutor-footer {
            text-align: center;
            margin-top: 1rem;
            font-size: 0.8rem;
            color: #666;
        }
        """
        
        # Создание Gradio интерфейса
        with gr.Blocks(css=css, title="ЕГЭ Математика - Тьютор") as self.interface:
            gr.HTML("""
            <div class="math-tutor-header">
                <h1>🧮 ЕГЭ Математика - Интерактивный тьютор</h1>
            </div>
            """)
            
            gr.Markdown("""
            <div class="math-tutor-description">
            Этот тьютор поможет вам подготовиться к ЕГЭ по математике. 
            Он не даст вам прямых ответов, а будет направлять вас к правильному решению.
            
            Вы можете:
            - Задать вопрос по математике
            - Отправить задачу для анализа
            - Показать свое решение для проверки
            - Попросить подсказку, если затрудняетесь
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=7):
                    chatbot = gr.Chatbot(
                        height=500,
                        show_label=False,
                        elem_id="math-tutor-chatbot"
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Введите ваш вопрос, задачу или решение...",
                            show_label=False,
                            elem_id="math-tutor-input"
                        )
                        submit_btn = gr.Button("Отправить", variant="primary")
                    
                    with gr.Row():
                        clear_btn = gr.Button("Очистить чат")
                        hint_btn = gr.Button("Подсказка")
                
                with gr.Column(scale=3):
                    with gr.Accordion("Загрузить изображение с задачей", open=False):
                        image_input = gr.Image(
                            type="pil",
                            label="Загрузите изображение с задачей"
                        )
                        upload_btn = gr.Button("Распознать и отправить")
                    
                    with gr.Accordion("Настройки тьютора", open=False):
                        hint_level = gr.Radio(
                            ["Легкая", "Средняя", "Сильная"],
                            label="Уровень подсказок",
                            value="Средняя"
                        )
                        
                        explanation_level = gr.Radio(
                            ["Базовый", "Средний", "Продвинутый"],
                            label="Уровень объяснений",
                            value="Средний"
                        )
                    
                    with gr.Accordion("Информация о ЕГЭ", open=False):
                        gr.Markdown("""
                        ### Структура ЕГЭ по математике (профильный уровень)
                        
                        - **Часть 1**: 12 заданий с кратким ответом (1 балл каждое)
                        - **Часть 2**: 7 заданий с развернутым ответом (2-4 балла каждое)
                        - **Всего**: 19 заданий, максимум 32 балла
                        
                        ### Основные разделы
                        
                        - Алгебра
                        - Геометрия
                        - Вероятность и статистика
                        
                        ### Время выполнения
                        
                        - 3 часа 55 минут (235 минут)
                        """)
            
            gr.HTML("""
            <div class="math-tutor-footer">
                <p>ИИ-тьютор по математике для подготовки к ЕГЭ | Создан с использованием LlamaIndex и Gradio</p>
            </div>
            """)
            
            # Обработчики событий
            msg.submit(self.chat, [msg, chatbot], [msg, chatbot])
            submit_btn.click(self.chat, [msg, chatbot], [msg, chatbot])
            clear_btn.click(self.clear_chat, [], [chatbot])
            hint_btn.click(self.get_hint, [chatbot, hint_level], [chatbot])
            upload_btn.click(self.process_image, [image_input, chatbot], [chatbot, image_input])
    
    def chat(self, message, history):
        """
        Обрабатывает сообщение пользователя и возвращает ответ агента.
        
        Args:
            message: Сообщение пользователя.
            history: История диалога.
            
        Returns:
            tuple: Обновленное сообщение и история диалога.
        """
        if not message:
            return "", history
        
        # Получение ответа от агента
        response = self.agent.chat(message)
        
        # Обновление истории
        history.append((message, response))
        
        return "", history
    
    def clear_chat(self):
        """
        Очищает историю чата.
        
        Returns:
            list: Пустая история чата.
        """
        # Очистка памяти агента
        self.agent.memory.clear()
        
        return []
    
    def get_hint(self, history, hint_level):
        """
        Генерирует подсказку для текущей задачи.
        
        Args:
            history: История диалога.
            hint_level: Уровень подсказки.
            
        Returns:
            list: Обновленная история диалога.
        """
        if not history:
            return history
        
        # Получение текущей задачи из истории
        current_problem = None
        for i in range(len(history) - 1, -1, -1):
            if "задача" in history[i][0].lower() or "задание" in history[i][0].lower():
                current_problem = history[i][0]
                break
        
        if not current_problem:
            # Если задача не найдена, запрашиваем ее
            message = "Пожалуйста, сначала отправьте задачу, для которой нужна подсказка."
            history.append((f"Запрос подсказки ({hint_level})", message))
            return history
        
        # Получение текущего прогресса из истории
        current_progress = ""
        for i in range(len(history) - 1, -1, -1):
            if "решение" in history[i][0].lower() or "мой ответ" in history[i][0].lower():
                current_progress = history[i][0]
                break
        
        # Преобразование уровня подсказки
        hint_level_map = {
            "Легкая": "light_hint",
            "Средняя": "medium_hint",
            "Сильная": "strong_hint"
        }
        
        # Генерация подсказки
        hint_request = f"Мне нужна {hint_level.lower()} подсказка для этой задачи."
        if current_progress:
            hint_request += f" Вот мой текущий прогресс: {current_progress}"
        
        # Получение ответа от агента
        response = self.agent.chat(hint_request)
        
        # Обновление истории
        history.append((f"Запрос подсказки ({hint_level})", response))
        
        return history
    
    def process_image(self, image, history):
        """
        Обрабатывает изображение с задачей.
        
        Args:
            image: Изображение с задачей.
            history: История диалога.
            
        Returns:
            tuple: Обновленная история диалога и пустое изображение.
        """
        if image is None:
            return history, None
        
        # Создание временного файла для изображения
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image_path = temp_file.name
            image.save(image_path)
        
        try:
            # Распознавание текста на изображении
            text = pytesseract.image_to_string(image, lang='rus')
            
            # Очистка распознанного текста
            text = text.strip()
            
            if not text:
                message = "Не удалось распознать текст на изображении. Пожалуйста, убедитесь, что изображение четкое и содержит текст."
                history.append(("Загрузка изображения", message))
                return history, None
            
            # Добавление распознанного текста в историю
            history.append(("Задача с изображения", text))
            
            # Получение ответа от агента
            response = self.agent.chat(f"Проанализируй эту задачу ЕГЭ по математике: {text}")
            
            # Обновление истории
            history.append(("Задача с изображения", response))
            
        except Exception as e:
            # В случае ошибки
            message = f"Произошла ошибка при обработке изображения: {str(e)}"
            history.append(("Загрузка изображения", message))
        
        finally:
            # Удаление временного файла
            if os.path.exists(image_path):
                os.remove(image_path)
        
        return history, None
    
    def launch(self, share=False):
        """
        Запускает Gradio интерфейс.
        
        Args:
            share: Флаг для создания публичной ссылки.
        """
        self.interface.launch(share=share)

# Создание и запуск приложения
if __name__ == "__main__":
    app = MathTutorApp()
    app.launch()
