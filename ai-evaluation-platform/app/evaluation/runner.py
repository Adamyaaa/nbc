import json
import statistics
from typing import List, Dict, Any
from app.services.rag import RAGService
from app.evaluation.judge import LLMJudge

class ExperimentRunner:
    def __init__(self, rag_service: RAGService, judge: LLMJudge):
        self.rag_service = rag_service
        self.judge = judge

    def run_experiment(self, dataset: List[Dict[str, str]], top_k: int = 5, model: str = None) -> Dict[str, Any]:
        """
        Runs the RAG pipeline over a dataset and evaluates the results.
        Dataset format expected: [{"question": "...", "expected_answer": "..."}, ...]
        """
        results = []
        metrics = {
            "correctness": [],
            "relevance": [],
            "groundedness": [],
            "latency": []
        }

        for item in dataset:
            question = item["question"]
            expected = item["expected_answer"]
            
            # 1. Run RAG
            answer = self.rag_service.answer_question(question, top_k=top_k, model=model)
            
            # 2. Reconstruct context string for the judge
            # In a real system, we'd pass the actual text from the retrieved chunks.
            # Here we just fetch them again or modify RAGService to return them.
            # For simplicity, we assume RAGService's retrieval can be queried again.
            context_obj = self.rag_service.retrieval_engine.retrieve(question, top_k=top_k)
            context_text = "\n".join([c.text for c in context_obj.chunks])
            
            # 3. Judge
            eval_res = self.judge.evaluate(
                question=question,
                expected_answer=expected,
                retrieved_context=context_text,
                generated_answer=answer.answer
            )
            
            # 4. Record
            metrics["correctness"].append(eval_res.correctness)
            metrics["relevance"].append(eval_res.relevance)
            metrics["groundedness"].append(eval_res.groundedness)
            metrics["latency"].append(answer.metadata.get("latency_seconds", 0.0))
            
            results.append({
                "question": question,
                "expected": expected,
                "generated": answer.answer,
                "evaluation": eval_res.model_dump(),
                "latency": answer.metadata.get("latency_seconds", 0.0)
            })
            
        summary = {
            "avg_correctness": statistics.mean(metrics["correctness"]) if metrics["correctness"] else 0.0,
            "avg_relevance": statistics.mean(metrics["relevance"]) if metrics["relevance"] else 0.0,
            "avg_groundedness": statistics.mean(metrics["groundedness"]) if metrics["groundedness"] else 0.0,
            "avg_latency": statistics.mean(metrics["latency"]) if metrics["latency"] else 0.0,
            "total_queries": len(dataset)
        }
        
        return {
            "summary": summary,
            "details": results
        }
        
    def save_report(self, report: Dict[str, Any], filepath: str = "experiment_report.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
