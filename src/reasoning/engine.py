import json
import urllib.error
import urllib.request
from .models import Answer


class ReasoningEngine:
    def answer(self, query, evidence) -> Answer:
        raise NotImplementedError


class RuleBasedFallbackReasoningEngine(ReasoningEngine):
    def answer(self, query, evidence=None, visual_observations=None):
        if hasattr(query, "retrieved_sources"):
            context = query
            query, evidence = context.question, context.retrieved_sources
            visual_observations = context.visual_observations
        evidence = evidence or []
        visual_observations = visual_observations or []
        if not evidence:
            return Answer(query, "Insufficient evidence in the local knowledge base or visual input.",
                          uncertainty="Visual similarity can suggest visible concepts but cannot establish a fault; no matching manual evidence was retrieved.",
                          observations=visual_observations)
        checks = []
        for item in evidence:
            text = " ".join(item.text.split())
            # Surface relevant source sentences as evidence-led checks, preserving wording.
            for sentence in text.replace("!", ".").split("."):
                sentence = sentence.strip()
                if "edgefix ai / synthetic training material" in sentence.lower() or sentence.lower().startswith("original synthetic content"):
                    continue
                sentence = sentence.replace("Original synthetic content · Not manufacturer guidance", "").strip()
                if len(sentence) > 30 and any(w in sentence.lower() for w in
                    ("inspect", "check", "verify", "measure", "clean", "confirm", "ensure", "replace", "lubric")):
                    if sentence not in checks:
                        checks.append(sentence)
                if len(checks) >= 5:
                    break
            if len(checks) >= 5:
                break
        observation = "No model-supported visual observation is available."
        if visual_observations:
            labels = ", ".join(str(item.get("label", "visual concept")) for item in visual_observations[:2])
            observation = f"The image is visually consistent with: {labels}. This is a similarity signal, not a confirmed equipment identification or diagnosis."
        return Answer(query, "Local manual evidence was retrieved. The cited passages guide the checks below; they do not confirm a fault.",
                      checks or ["Review the cited maintenance guidance and compare it with the observed condition."], sources=evidence,
                      uncertainty="The image and retrieved passages do not establish the cause. Confirm observations using approved site methods.",
                      observations=visual_observations, evidence_strength="combined" if visual_observations else "documentation")


class OllamaReasoningEngine(ReasoningEngine):
    def __init__(self, url, model, timeout=25):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.fallback = RuleBasedFallbackReasoningEngine()
        self.available = self._check()

    def _check(self):
        try:
            with urllib.request.urlopen(self.url + "/api/tags", timeout=1) as response:
                tags = json.loads(response.read())
            return any(x.get("name", "").split(":")[0] == self.model.split(":")[0] for x in tags.get("models", []))
        except Exception:
            return False

    def answer(self, query, evidence=None, visual_observations=None):
        if hasattr(query, "retrieved_sources"):
            context = query
            query, evidence = context.question, context.retrieved_sources
            visual_observations = context.visual_observations
        evidence = evidence or []
        visual_observations = visual_observations or []
        if not evidence:
            return self.fallback.answer(query, evidence, visual_observations)
        sources = "\n".join(f"[{i+1}] {x.document_name}, p.{x.page_number}, {x.section}: {x.text}" for i, x in enumerate(evidence))
        visual = "\n".join(f"{x.get('label')}: relative cosine similarity {x.get('similarity', 'not reported')}" for x in visual_observations[:5]) or "Unavailable"
        prompt = ("You are EdgeFix, a local maintenance assistant. Ground claims only in these visual similarity observations and retrieved excerpts, plus plainly labeled general safety guidance. "
                  "Do not treat CLIP similarity as a diagnosis. Never invent manual names, sections, page numbers, or citations. "
                  "If support is insufficient, include exactly: Insufficient evidence in the local knowledge base or visual input. "
                  "Return sections OBSERVATION, EVIDENCE, RECOMMENDED CHECKS, SAFETY, UNCERTAINTY.\n"
                  f"Question: {query}\nVisual similarity observations:\n{visual}\nRetrieved manual evidence:\n{sources}")
        try:
            request = urllib.request.Request(self.url + "/api/generate", data=json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read()).get("response", "").strip()
            if not result:
                raise ValueError("Ollama returned an empty response")
            return Answer(query, result, ["See the cited evidence and generated answer."], sources=evidence, mode=f"OLLAMA · {self.model}",
                          uncertainty="Generated locally from retrieved excerpts; verify against approved site procedures.",
                          observations=visual_observations, evidence_strength="combined" if visual_observations else "documentation")
        except Exception:
            return self.fallback.answer(query, evidence, visual_observations)


def build_reasoner(settings):
    engine = OllamaReasoningEngine(settings.ollama_url, settings.ollama_model)
    return engine if engine.available else RuleBasedFallbackReasoningEngine()
