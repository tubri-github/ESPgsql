# 搜索功能文档

本文档描述了 ESPgsql API 中实现的搜索功能。

## 概述

搜索 API 提供多种搜索策略来查找 Elasticsearch 中的物种记录：
- **精确搜索**：标准词条匹配
- **模糊搜索**：自动编辑距离处理拼写错误
- **语音搜索**：查找发音相似的名称（sounds-like 匹配）
- **通配符搜索**：使用 `*` 和 `?` 运算符进行模式匹配
- **同义名解析**：通过 ValidName 字段自动将同义名解析为有效名

## 搜索端点

### GET /search

主搜索端点，具有智能回退逻辑。

**参数：**
- `scientificname` (str)：要搜索的物种名称
- `family` (str)：科名过滤器
- `country` (str)：国家过滤器
- `institutioncode` (str)：机构代码过滤器
- `size` (int, 默认=50)：返回结果数量
- `from_` (int, 默认=0)：分页偏移量

**搜索行为：**
1. 检测查询是否包含通配符（`*` 或 `?`）
2. 如果有通配符：使用通配符查询
3. 否则：使用智能回退搜索（精确 -> 如果结果 < 阈值则模糊）

### GET /search/phonetic

使用 double_metaphone 算法的语音搜索。

**工作原理：**
- 使用 Elasticsearch 的 analysis-phonetic 插件
- 搜索 ScientificName、Family、Genus 和 ValidName 的 `.phonetic` 子字段
- 适合在知道发音但不确定准确拼写时查找物种

### GET /search/fuzzy

可配置模糊度的模糊搜索。

**参数：**
- `fuzziness` (str, 默认="AUTO")：模糊度级别
  - "AUTO"：基于词条长度的编辑距离（推荐）
  - "0", "1", "2"：固定编辑距离

### GET /search/wildcard

基于模式的搜索。

**通配符运算符：**
- `*`：匹配任意字符序列（包括空）
- `?`：精确匹配一个字符

**示例：**
- `Gadus*` - 匹配 "Gadus morhua"、"Gadus macrocephalus" 等
- `Gad?s` - 匹配 "Gadus"、"Gadis" 等
- `*morhua` - 匹配任何以 "morhua" 结尾的名称

## 同义名解析

### 工作原理

系统使用在 ES 索引同步期间填充的 `ValidName` 字段：

1. 索引期间（`sync_es_index.py`）：
   - 每条记录的 ScientificName 在 TaxonRank 数据库中查找
   - 如果是同义名，则检索有效名
   - ScientificName（原始）和 ValidName（解析后）都被索引

2. 搜索期间：
   - 查询同时搜索 ScientificName 和 ValidName 字段
   - 查找原始名称或有效名称都会返回匹配的记录

### 示例

如果 "Gadus callarias" 是 "Gadus morhua" 的同义名：
- 搜索 "Gadus callarias" 会找到 `ScientificName="Gadus callarias"` 或 `ValidName="Gadus callarias"` 的记录
- 有效名解析确保同义名查询返回相同的标本

## 文档标准化

所有搜索响应都经过文档标准化以确保一致的字段命名：

**标准化字段：**
- `CatalogNumber`：唯一标本标识符
- `ScientificName`：物种名称
- `Family`：分类科
- `ValidName`：解析后的有效名（如果是同义名）
- `Country`、`Locality`：地理信息
- `Latitude`、`Longitude`：坐标
- `InstitutionCode`、`CollectionCode`：来源机构
- `YearCollected`、`MonthCollected`、`DayCollected`：采集日期
- 等等...

## 技术实现

### Elasticsearch 索引映射

关键字段配置：
- `ScientificName`：text + keyword + phonetic 子字段
- `Family`：text + keyword + phonetic 子字段
- `ValidName`：text + keyword + phonetic 子字段
- `Genus`：text + keyword + phonetic 子字段
- 地理/数值字段：适当的类型（float、long、keyword）

### 语音分析器

```json
{
  "analysis": {
    "filter": {
      "phonetic_filter": {
        "type": "phonetic",
        "encoder": "double_metaphone",
        "replace": false
      }
    },
    "analyzer": {
      "phonetic_analyzer": {
        "type": "custom",
        "tokenizer": "standard",
        "filter": ["lowercase", "phonetic_filter"]
      }
    }
  }
}
```

### 智能回退逻辑

```python
# 伪代码
def smart_search(query):
    results = exact_search(query)
    if len(results) < threshold:  # 默认阈值：5
        fuzzy_results = fuzzy_search(query, fuzziness="AUTO")
        if len(fuzzy_results) > len(results):
            return fuzzy_results
    return results
```

## 环境要求

- Elasticsearch 8.x 并安装 analysis-phonetic 插件
- PostgreSQL 包含 harvestedfn2 表
- TaxonRank 数据库用于同义名解析

## 另请参阅

- `sync_es_index.py`：带有 ValidName 填充的 ES 索引同步脚本
- `main.py`：包含搜索端点的 FastAPI 应用程序
- `docs/ES_SYNONYM_UPGRADE_PLAN.md`：未来 ES 同义词过滤器升级计划
