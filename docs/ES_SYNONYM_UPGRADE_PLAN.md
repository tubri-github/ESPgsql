# ES 同义词索引升级方案（方案2）

> 本文档记录将来升级到 ES 原生同义词支持的实现方案

## 概述

使用 Elasticsearch 的 synonym filter 在索引层面处理同义名，实现最佳查询性能。

## 实现步骤

### 1. 从 TaxonRank 导出同义词文件

创建脚本 `export_synonyms.py`：

```python
"""
从 TaxonRank 数据库导出 ES 同义词文件
"""
import psycopg2

# TaxonRank 数据库配置
TAXON_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "taxonomy_dev",
    "user": "postgres",
    "password": "your_password"
}

def export_synonyms(output_file="taxon_synonyms.txt"):
    conn = psycopg2.connect(**TAXON_DB_CONFIG)
    cursor = conn.cursor()

    # 查询所有同义名关系
    query = """
        SELECT
            v.scientific_name as valid_name,
            s.scientific_name as synonym_name
        FROM taxa v
        JOIN taxa s ON s.valid_id = v.id
        WHERE v.status = 'valid' AND s.status = 'synonym'
        ORDER BY v.scientific_name
    """

    cursor.execute(query)

    # 按有效名分组
    synonyms_map = {}
    for valid_name, synonym_name in cursor.fetchall():
        if valid_name not in synonyms_map:
            synonyms_map[valid_name] = []
        synonyms_map[valid_name].append(synonym_name)

    # 写入 ES 同义词格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for valid_name, synonyms in synonyms_map.items():
            # 格式: synonym1, synonym2, synonym3 => valid_name
            if synonyms:
                line = f"{', '.join(synonyms)} => {valid_name}\n"
                f.write(line)

    cursor.close()
    conn.close()
    print(f"导出完成: {len(synonyms_map)} 个有效名, 共 {sum(len(s) for s in synonyms_map.values())} 个同义名")

if __name__ == "__main__":
    export_synonyms()
```

### 2. 配置 ES 同义词分析器

将同义词文件放到 ES 配置目录：
```
elasticsearch/config/analysis/taxon_synonyms.txt
```

### 3. 创建带同义词的索引 Mapping

```json
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "analysis": {
      "filter": {
        "taxon_synonym_filter": {
          "type": "synonym",
          "synonyms_path": "analysis/taxon_synonyms.txt",
          "updateable": true
        }
      },
      "analyzer": {
        "taxon_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": [
            "lowercase",
            "taxon_synonym_filter"
          ]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "ScientificName": {
        "type": "text",
        "analyzer": "taxon_analyzer",
        "fields": {
          "keyword": {
            "type": "keyword",
            "ignore_above": 256
          },
          "raw": {
            "type": "text",
            "analyzer": "standard"
          }
        }
      }
    }
  }
}
```

### 4. 搜索时自动应用同义词

```python
# 搜索 "Gadus callarias" 会自动匹配 "Gadus morhua" 的记录
query = {
    "query": {
        "match": {
            "ScientificName": "Gadus callarias"
        }
    }
}
```

## 同义词更新流程

1. 重新运行 `export_synonyms.py` 导出最新数据
2. 替换 ES 配置目录中的同义词文件
3. 重建索引（或使用 ES 7.3+ 的 reload API）

```bash
# ES 7.3+ 可以动态重载同义词（需要 updateable: true）
POST /mrservice_harvestedfn2_index/_reload_search_analyzers
```

## 优缺点

**优点：**
- 查询性能最佳，ES 原生支持
- 搜索语法简单，无需额外处理
- 支持模糊匹配 + 同义词组合

**缺点：**
- 同义词更新后需要重载/重建索引
- 同义词文件需要维护
- 大量同义词可能影响索引速度

## 参考资料

- [ES Synonym Token Filter](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html)
- [ES Synonym Graph Token Filter](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-graph-tokenfilter.html)
- [Updateable Synonym Filters](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-analyzer.html)

---

**预计升级时间**: 待定
**当前方案**: 方案3（数据预处理，ValidName 字段）