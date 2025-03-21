"""
Векторное хранилище Chroma для базы заданий ЕГЭ по математике и их решений.
"""

from typing import List, Dict, Any, Optional
import os
import json
from pathlib import Path
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode, IndexNode
import chromadb

class MathProblemVectorStore:
    """
    Векторное хранилище для базы заданий ЕГЭ по математике и их решений.
    
    Это хранилище использует Chroma для хранения и поиска задач ЕГЭ по математике,
    их решений и связанных метаданных.
    """
    
    def __init__(self, persist_dir: str = "./data/chroma_db"):
        """
        Инициализация векторного хранилища.
        
        Args:
            persist_dir: Директория для хранения векторной базы данных.
        """
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        # Инициализация клиента Chroma
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Создание коллекций
        self.problems_collection = self.chroma_client.get_or_create_collection("math_problems")
        self.solutions_collection = self.chroma_client.get_or_create_collection("math_solutions")
        self.concepts_collection = self.chroma_client.get_or_create_collection("math_concepts")
        
        # Создание векторных хранилищ
        self.problems_vector_store = ChromaVectorStore(chroma_collection=self.problems_collection)
        self.solutions_vector_store = ChromaVectorStore(chroma_collection=self.solutions_collection)
        self.concepts_vector_store = ChromaVectorStore(chroma_collection=self.concepts_collection)
        
        # Создание контекстов хранения
        self.problems_storage_context = StorageContext.from_defaults(vector_store=self.problems_vector_store)
        self.solutions_storage_context = StorageContext.from_defaults(vector_store=self.solutions_vector_store)
        self.concepts_storage_context = StorageContext.from_defaults(vector_store=self.concepts_vector_store)
        
        # Создание индексов
        self.problems_index = VectorStoreIndex.from_documents(
            [], storage_context=self.problems_storage_context
        )
        self.solutions_index = VectorStoreIndex.from_documents(
            [], storage_context=self.solutions_storage_context
        )
        self.concepts_index = VectorStoreIndex.from_documents(
            [], storage_context=self.concepts_storage_context
        )
    
    def add_problem(self, problem_id: str, problem_text: str, metadata: Dict[str, Any]) -> None:
        """
        Добавляет задачу в векторное хранилище.
        
        Args:
            problem_id: Идентификатор задачи.
            problem_text: Текст задачи.
            metadata: Метаданные задачи (тип, сложность, ключевые концепции и т.д.).
        """
        # Создание документа
        document = Document(
            text=problem_text,
            metadata={
                "problem_id": problem_id,
                **metadata
            }
        )
        
        # Добавление документа в индекс
        self.problems_index.insert(document)
    
    def add_solution(self, problem_id: str, solution_text: str, metadata: Dict[str, Any]) -> None:
        """
        Добавляет решение задачи в векторное хранилище.
        
        Args:
            problem_id: Идентификатор задачи.
            solution_text: Текст решения.
            metadata: Метаданные решения (шаги, формулы, ключевые моменты и т.д.).
        """
        # Создание документа
        document = Document(
            text=solution_text,
            metadata={
                "problem_id": problem_id,
                **metadata
            }
        )
        
        # Добавление документа в индекс
        self.solutions_index.insert(document)
    
    def add_concept(self, concept_id: str, concept_name: str, concept_description: str, metadata: Dict[str, Any]) -> None:
        """
        Добавляет математическую концепцию в векторное хранилище.
        
        Args:
            concept_id: Идентификатор концепции.
            concept_name: Название концепции.
            concept_description: Описание концепции.
            metadata: Метаданные концепции (формулы, примеры, связанные концепции и т.д.).
        """
        # Создание документа
        document = Document(
            text=f"{concept_name}: {concept_description}",
            metadata={
                "concept_id": concept_id,
                "concept_name": concept_name,
                **metadata
            }
        )
        
        # Добавление документа в индекс
        self.concepts_index.insert(document)
    
    def search_problems(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Ищет задачи в векторном хранилище.
        
        Args:
            query: Запрос для поиска.
            top_k: Количество результатов.
            filters: Фильтры для поиска (тип задачи, сложность и т.д.).
            
        Returns:
            List[Dict[str, Any]]: Список найденных задач.
        """
        # Создание поискового движка
        retriever = self.problems_index.as_retriever(similarity_top_k=top_k)
        
        # Поиск задач
        nodes = retriever.retrieve(query)
        
        # Фильтрация результатов
        if filters:
            filtered_nodes = []
            for node in nodes:
                match = True
                for key, value in filters.items():
                    if key not in node.metadata or node.metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_nodes.append(node)
            nodes = filtered_nodes
        
        # Форматирование результатов
        results = []
        for node in nodes:
            results.append({
                "problem_id": node.metadata.get("problem_id", ""),
                "text": node.text,
                "metadata": node.metadata,
                "score": node.score
            })
        
        return results
    
    def search_solutions(self, query: str, top_k: int = 5, problem_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Ищет решения в векторном хранилище.
        
        Args:
            query: Запрос для поиска.
            top_k: Количество результатов.
            problem_id: Идентификатор задачи (если нужны решения конкретной задачи).
            
        Returns:
            List[Dict[str, Any]]: Список найденных решений.
        """
        # Создание поискового движка
        retriever = self.solutions_index.as_retriever(similarity_top_k=top_k)
        
        # Поиск решений
        nodes = retriever.retrieve(query)
        
        # Фильтрация по идентификатору задачи
        if problem_id:
            nodes = [node for node in nodes if node.metadata.get("problem_id") == problem_id]
        
        # Форматирование результатов
        results = []
        for node in nodes:
            results.append({
                "problem_id": node.metadata.get("problem_id", ""),
                "text": node.text,
                "metadata": node.metadata,
                "score": node.score
            })
        
        return results
    
    def search_concepts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Ищет математические концепции в векторном хранилище.
        
        Args:
            query: Запрос для поиска.
            top_k: Количество результатов.
            
        Returns:
            List[Dict[str, Any]]: Список найденных концепций.
        """
        # Создание поискового движка
        retriever = self.concepts_index.as_retriever(similarity_top_k=top_k)
        
        # Поиск концепций
        nodes = retriever.retrieve(query)
        
        # Форматирование результатов
        results = []
        for node in nodes:
            results.append({
                "concept_id": node.metadata.get("concept_id", ""),
                "concept_name": node.metadata.get("concept_name", ""),
                "text": node.text,
                "metadata": node.metadata,
                "score": node.score
            })
        
        return results
    
    def get_problem_by_id(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """
        Получает задачу по идентификатору.
        
        Args:
            problem_id: Идентификатор задачи.
            
        Returns:
            Optional[Dict[str, Any]]: Задача или None, если задача не найдена.
        """
        # Поиск задачи по идентификатору
        nodes = self.problems_collection.get(where={"problem_id": problem_id})
        
        if not nodes or not nodes.get("ids"):
            return None
        
        # Получение первого результата
        node_id = nodes["ids"][0]
        node_text = nodes["documents"][0]
        node_metadata = json.loads(nodes["metadatas"][0])
        
        return {
            "problem_id": problem_id,
            "text": node_text,
            "metadata": node_metadata
        }
    
    def get_solutions_by_problem_id(self, problem_id: str) -> List[Dict[str, Any]]:
        """
        Получает решения задачи по идентификатору задачи.
        
        Args:
            problem_id: Идентификатор задачи.
            
        Returns:
            List[Dict[str, Any]]: Список решений.
        """
        # Поиск решений по идентификатору задачи
        nodes = self.solutions_collection.get(where={"problem_id": problem_id})
        
        if not nodes or not nodes.get("ids"):
            return []
        
        # Форматирование результатов
        results = []
        for i in range(len(nodes["ids"])):
            results.append({
                "solution_id": nodes["ids"][i],
                "problem_id": problem_id,
                "text": nodes["documents"][i],
                "metadata": json.loads(nodes["metadatas"][i])
            })
        
        return results
    
    def load_problems_from_json(self, json_file: str) -> int:
        """
        Загружает задачи из JSON-файла.
        
        Args:
            json_file: Путь к JSON-файлу с задачами.
            
        Returns:
            int: Количество загруженных задач.
        """
        # Проверка существования файла
        if not os.path.exists(json_file):
            return 0
        
        # Загрузка данных из файла
        with open(json_file, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        
        # Добавление задач в хранилище
        count = 0
        for problem in problems:
            problem_id = problem.get("problem_id", str(count))
            problem_text = problem.get("text", "")
            metadata = problem.get("metadata", {})
            
            self.add_problem(problem_id, problem_text, metadata)
            
            # Добавление решений, если они есть
            solutions = problem.get("solutions", [])
            for solution in solutions:
                solution_text = solution.get("text", "")
                solution_metadata = solution.get("metadata", {})
                
                self.add_solution(problem_id, solution_text, solution_metadata)
            
            count += 1
        
        return count
    
    def load_concepts_from_json(self, json_file: str) -> int:
        """
        Загружает математические концепции из JSON-файла.
        
        Args:
            json_file: Путь к JSON-файлу с концепциями.
            
        Returns:
            int: Количество загруженных концепций.
        """
        # Проверка существования файла
        if not os.path.exists(json_file):
            return 0
        
        # Загрузка данных из файла
        with open(json_file, 'r', encoding='utf-8') as f:
            concepts = json.load(f)
        
        # Добавление концепций в хранилище
        count = 0
        for concept in concepts:
            concept_id = concept.get("concept_id", str(count))
            concept_name = concept.get("name", "")
            concept_description = concept.get("description", "")
            metadata = concept.get("metadata", {})
            
            self.add_concept(concept_id, concept_name, concept_description, metadata)
            
            count += 1
        
        return count
    
    def export_problems_to_json(self, json_file: str) -> int:
        """
        Экспортирует задачи в JSON-файл.
        
        Args:
            json_file: Путь к JSON-файлу для экспорта.
            
        Returns:
            int: Количество экспортированных задач.
        """
        # Получение всех задач
        nodes = self.problems_collection.get()
        
        if not nodes or not nodes.get("ids"):
            return 0
        
        # Форматирование результатов
        problems = []
        for i in range(len(nodes["ids"])):
            problem_id = nodes["ids"][i]
            problem_text = nodes["documents"][i]
            metadata = json.loads(nodes["metadatas"][i])
            
            # Получение решений для задачи
            solutions = self.get_solutions_by_problem_id(problem_id)
            
            problems.append({
                "problem_id": problem_id,
                "text": problem_text,
                "metadata": metadata,
                "solutions": solutions
            })
        
        # Сохранение данных в файл
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(problems, f, ensure_ascii=False, indent=2)
        
        return len(problems)
