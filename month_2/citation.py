import re 

def verify(ans, chunks):
    c_ns = re.findall(r'\[(\d+)\]', ans)
    c_ns = (int(n) for n in c_ns)

    issues = []
    
    if not c_ns:
        return {
            "verified": False,
            "issues": ["No citations found in answer"],
            "answer": ans
        }
    
    for n in c_ns:
        idx = n - 1  
        
        if idx >= len(chunks):
            issues.append(f"Citation [{n}] has no corresponding source chunk")
            continue
        
        chunk = chunks[idx]["document"]
        chunk_meta = chunks[idx].get("metadata", {})
         
        
        source = f"{chunk_meta.get('source', 'unknown')}"
        if chunk_meta.get("page"):
            source += f", p.{chunk_meta['page']}"


    return {
        "verified": len(issues) == 0,
        "issues": issues,
        "citation_count": len(c_ns),
        "answer": ans
    }




