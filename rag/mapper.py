import json
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

from .retriever import multi_search
from .agents import _quick_llm, _extract_json_array, _dedupe_queries

# Logger setup
logger = logging.getLogger(__name__)

@dataclass
class ArgumentNode:
    id: str
    type: str  # "root", "claim", "objection", "rebuttal", "definition"
    content: str
    detailed_body: str = ""
    sources: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    children: List["ArgumentNode"] = field(default_factory=list)
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Recursive dictionary conversion for JSON export."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "detailed_body": self.detailed_body,
            "sources": self.sources,
            "relevance_score": self.relevance_score,
            "children": [child.to_dict() for child in self.children],
            "parent_id": self.parent_id
        }

class TopicMapper:
    def __init__(self, topic: str, max_depth: int = 3, max_children: int = 3):
        self.topic = topic
        self.max_depth = max_depth
        self.max_children = max_children
        self.root: Optional[ArgumentNode] = None
        self.visited_contents = set()

    def build_map(self) -> ArgumentNode:
        """Builds the argument map starting from the root topic."""
        print(f"🗺️  Mapping topic: {self.topic}")
        
        # 1. Create Root Node
        root_context = self._retrieve_context(self.topic)
        definition = self._define_topic(self.topic, root_context)
        
        self.root = ArgumentNode(
            id="root",
            type="root",
            content=f"{self.topic}: {definition}",
            detailed_body=root_context[:500] + "...",
            sources=[], # Root sources are general
            relevance_score=1.0
        )
        self.visited_contents.add(self.topic.lower())

        # 2. Start Recursive Expansion
        self._expand_node(self.root, current_depth=0)
        
        return self.root

    def _expand_node(self, node: ArgumentNode, current_depth: int):
        """Recursively finds counter-arguments and sub-claims."""
        if current_depth >= self.max_depth:
            return

        print(f"  🔍 Depth {current_depth}: Expanding '{node.content[:30]}...'")

        # Generate search queries based on the node's content
        # If it's a claim, look for objections. If objection, look for rebuttals.
        search_query = f"{node.content} eleştiri karşıt görüş itiraz"
        if node.type == "objection":
             search_query = f"{node.content} yanıt savunma cevap"

        # Search for counter-moves
        context_docs = self._retrieve_context(search_query)
        if not context_docs:
            return

        # Extract moves from context
        moves = self._extract_moves(node.content, context_docs, node.type)
        
        for move in moves[:self.max_children]:
            # Deduplication check
            if move["summary"].lower() in self.visited_contents:
                continue
            
            self.visited_contents.add(move["summary"].lower())
            
            new_type = "objection" if node.type in ["root", "claim", "rebuttal"] else "rebuttal"
            
            child_node = ArgumentNode(
                id=str(uuid.uuid4())[:8],
                type=new_type,
                content=move["summary"],
                detailed_body=move["detail"],
                sources=[], # Could attach specific chunk IDs if available
                relevance_score=0.9, # Placeholder
                parent_id=node.id
            )
            
            node.children.append(child_node)
            
            # Recurse
            self._expand_node(child_node, current_depth + 1)

    def _retrieve_context(self, query: str) -> str:
        """Wrapper for retriever.multi_search returning consolidated text."""
        docs = multi_search([query], top_k=5)
        return "\n\n".join([d.get("content", "") for d in docs])

    def _define_topic(self, topic: str, context: str) -> str:
        """Extracts a one-sentence definition of the topic."""
        prompt = f"""Konu: {topic}
Bağlam: {context[:2000]}

Bu konuyu tek, net, açıklayıcı bir cümle ile tanımla."""
        return _quick_llm(prompt, max_tokens=60)


    def _extract_moves(self, argument: str, context: str, current_type: str) -> List[Dict]:
        """Uses LLM to extract counter-arguments/moves from context."""
        
        target_type = "itiraz/eleştiri" if current_type in ["root", "claim", "rebuttal"] else "cevap/savunma"
        
        prompt = f"""Analiz edilecek argüman: "{argument}"

Aşağıdaki metinlerden, bu argümana yönelik TOP {self.max_children} adet farklı {target_type} çıkar.
Sadece metinde geçen gerçek felsefi görüşleri kullan. Uydurma.

Yanıtı şu JSON formatında ver:
[
  {{
    "summary": "Kısa, çarpıcı başlık (örn: Özgür İrade Savunması)",
    "detail": "Biraz daha detaylı açıklama (1-2 cümle)"
  }}
]

Metin Bağlamı:
{context[:4000]}"""

        result = _quick_llm(prompt, max_tokens=500)
        
        # Manuel extraction because _extract_json_array flattens dictionaries to strings
        cleaned = result.replace("```json", "").replace("```", "").strip()
        try:
             # Try simple parse first
             data = json.loads(cleaned)
             if isinstance(data, list) and all(isinstance(x, dict) for x in data):
                 return data
        except json.JSONDecodeError:
             pass
        
        # Fallback regex extraction for list of dicts
        import re
        pattern = r'\{[^{}]*"summary":\s*"([^"]+)",\s*"detail":\s*"([^"]+)"[^{}]*\}'
        found = []
        for match in re.finditer(pattern, cleaned, re.DOTALL):
            found.append({"summary": match.group(1), "detail": match.group(2)})
            
        return found

def export_markdown(node: ArgumentNode) -> str:
    """Generates a Markdown report with Mermaid diagram."""
    
    # 1. Mermaid Graph
    mermaid_lines = ["graph TD"]
    # Node definitions
    nodes = []
    edges = []
    
    def collect_nodes(n: ArgumentNode):
        # Escape content for mermaid
        safe_content = n.content.replace('"', "'").replace("(", "").replace(")", "")[:40]
        if len(n.content) > 40: safe_content += "..."
        
        # Shape logic
        if n.type == "root":
            shape = f'["{safe_content}"]'
        elif n.type == "objection":
            shape = f'("{safe_content}")' # Round edges
        else:
            shape = f'{{"{safe_content}"}}' # Hex/Rhombus approx
            
        nodes.append(f'    {n.id}{shape}')
        
        for child in n.children:
            edges.append(f'    {n.id} --> {child.id}')
            collect_nodes(child)
            
    collect_nodes(node)
    
    mermaid_block = "```mermaid\n" + "\n".join(mermaid_lines + nodes + edges) + "\n```"
    
    # 2. Detailed Text
    text_content = []
    
    def traverse_text(n: ArgumentNode, level: int):
        indent = "#" * min(level + 1, 6)
        icon = "🌳" if n.type == "root" else ("🔴" if n.type == "objection" else "🟢")
        
        text_content.append(f"{indent} {icon} {n.content}")
        text_content.append(f"_{n.type.upper()}_ | Score: {n.relevance_score}")
        text_content.append(f"\n{n.detailed_body}\n")
        
        if n.sources:
             text_content.append(f"> Kaynaklar: {', '.join(n.sources)}\n")
        
        for child in n.children:
            traverse_text(child, level + 1)

    traverse_text(node, 1)
    
    return f"# Felsefi Argüman Haritası\n\n## Görsel Özet\n\n{mermaid_block}\n\n## Detaylı Akış\n\n" + "\n".join(text_content)

def export_interactive_html(node: ArgumentNode) -> str:
    """Generates a standalone HTML file with interactive vis.js graph."""
    
    nodes_data = []
    edges_data = []
    
    def traverse(n: ArgumentNode):
        # Color coding
        color = "#FFD700"  # Gold for Root
        if n.type == "objection":
            color = "#FF6B6B"  # Soft Red
        elif n.type == "rebuttal":
            color = "#4ECDC4"  # Soft Teal/Green
            
        # Shape
        shape = "star" if n.type == "root" else "dot"
        size = 30 if n.type == "root" else 20
        
        # Tooltip
        title = f"<b>{n.type.upper()}</b><br>{n.content}"
        
        nodes_data.append({
            "id": n.id,
            "label": n.content[:20] + "...",
            "title": title,
            "color": color,
            "shape": shape,
            "size": size,
            "font": {"multi": "html", "size": 14},
            # Hidden data for sidebar
            "full_content": n.content,
            "detailed_body": n.detailed_body.replace("\n", "<br>"),
            "type": n.type.upper(),
            "score": n.relevance_score,
            "sources": n.sources
        })
        
        for child in n.children:
            edges_data.append({"from": n.id, "to": child.id, "arrows": "to"})
            traverse(child)
            
    traverse(node)
    
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PhilAI Argument Map</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; overflow: hidden; }}
        #network {{ width: 100vw; height: 100vh; background: #222; }}
        #sidebar {{
            position: absolute; top: 0; right: -400px; width: 350px; height: 100%;
            background: rgba(255, 255, 255, 0.95); box-shadow: -2px 0 10px rgba(0,0,0,0.5);
            transition: right 0.3s ease; padding: 20px; box-sizing: border-box; overflow-y: auto;
        }}
        #sidebar.open {{ right: 0; }}
        .close-btn {{ float: right; cursor: pointer; font-size: 20px; font-weight: bold; }}
        h2 {{ margin-top: 0; color: #333; }}
        .badge {{ 
            display: inline-block; padding: 4px 8px; border-radius: 4px; 
            font-size: 12px; font-weight: bold; color: white; margin-bottom: 10px;
        }}
        .content {{ font-size: 16px; line-height: 1.5; color: #444; }}
        .details {{ margin-top: 20px; font-size: 14px; color: #666; border-top: 1px solid #ddd; padding-top: 20px; }}
        .sources {{ margin-top: 10px; font-size: 12px; color: #888; font-style: italic; }}
    </style>
</head>
<body>
    <div id="network"></div>
    <div id="sidebar">
        <span class="close-btn" onclick="closeSidebar()">×</span>
        <div id="node-info">
            <h2 id="node-type">Node Type</h2>
            <div id="node-badge" class="badge">TYPE</div>
            <div id="node-content" class="content">Content goes here...</div>
            <div id="node-details" class="details">Details...</div>
            <div id="node-sources" class="sources"></div>
        </div>
    </div>

    <script type="text/javascript">
        // Data injected from Python
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});

        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            nodes: {{ borderWidth: 2, shadow: true }},
            edges: {{ width: 2, shadow: true, smooth: {{ type: "cubicBezier" }} }},
            layout: {{ hierarchical: {{ direction: "UD", sortMethod: "directed", levelSeparation: 150, nodeSpacing: 200 }} }},
            physics: {{ hierarchicalRepulsion: {{ nodeDistance: 200 }} }}
        }};
        
        var network = new vis.Network(container, data, options);
        
        network.on("click", function (params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                showSidebar(node);
            }} else {{
                closeSidebar();
            }}
        }});
        
        function showSidebar(node) {{
            document.getElementById('sidebar').classList.add('open');
            document.getElementById('node-type').innerText = node.type;
            
            var badge = document.getElementById('node-badge');
            badge.innerText = node.type;
            badge.style.backgroundColor = node.color;
            
            document.getElementById('node-content').innerText = node.full_content;
            document.getElementById('node-details').innerHTML = node.detailed_body;
            
            if (node.sources && node.sources.length > 0) {{
                document.getElementById('node-sources').innerText = "Kaynaklar: " + node.sources.join(", ");
            }} else {{
                document.getElementById('node-sources').innerText = "";
            }}
        }}
        
        function closeSidebar() {{
            document.getElementById('sidebar').classList.remove('open');
        }}
    </script>
</body>
</html>
"""
    return html_template

def export_json(node: ArgumentNode) -> str:
    return json.dumps(node.to_dict(), ensure_ascii=False, indent=2)
