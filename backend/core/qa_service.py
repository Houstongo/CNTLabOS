"""
QA Chat Service for CNTA Research Assistant

专门为科研问答场景设计的对话服务，集成RAG知识库检索
"""

import json
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime


class QAService:
    """科研问答服务"""

    def __init__(self, kb_db_path: str):
        """
        初始化QA服务

        Args:
            kb_db_path: 知识库数据库路径
        """
        self.kb_db_path = kb_db_path

    def _connect(self):
        """创建数据库连接"""
        conn = sqlite3.connect(self.kb_db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_conversations(
        self, session_id: str, limit: int = 50
    ) -> List[Dict]:
        """
        获取对话历史

        Args:
            session_id: 会话标识
            limit: 返回的最大条数

        Returns:
            对话记录列表，按时间倒序排列
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, rag_context, sources, created_at
                FROM kb_conversations
                WHERE session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def save_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
        rag_context: Optional[Dict] = None,
        sources: Optional[List] = None,
    ) -> int:
        """
        保存对话记录

        Args:
            session_id: 会话标识
            role: 'user' 或 'assistant'
            content: 消息内容
            rag_context: RAG上下文摘要（字典）
            sources: 引用来源列表

        Returns:
            新增记录的ID
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO kb_conversations (session_id, role, content, rag_context, sources)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    json.dumps(rag_context, ensure_ascii=False) if rag_context else None,
                    json.dumps(sources, ensure_ascii=False) if sources else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def delete_conversation(self, conv_id: int) -> None:
        """
        删除单条对话

        Args:
            conv_id: 对话记录ID
        """
        conn = self._connect()
        try:
            conn.execute("DELETE FROM kb_conversations WHERE id = ?", (conv_id,))
            conn.commit()
        finally:
            conn.close()

    def clear_session(self, session_id: str) -> int:
        """
        清空会话

        Args:
            session_id: 会话标识

        Returns:
            删除的记录数
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM kb_conversations WHERE session_id = ?", (session_id,)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_templates(self) -> List[Dict]:
        """
        获取预设问题模板

        Returns:
            模板列表，按分类分组
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, category, question, icon, created_at
                FROM kb_qa_templates
                ORDER BY category, id
                """
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def add_template(
        self, category: str, question: str, icon: Optional[str] = None
    ) -> int:
        """
        添加预设问题模板

        Args:
            category: 分类（文献/工艺/形貌/性能）
            question: 问题文本
            icon: 图标名称（可选）

        Returns:
            新增模板的ID
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO kb_qa_templates (category, question, icon)
                VALUES (?, ?, ?)
                """,
                (category, question, icon),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def delete_template(self, template_id: int) -> None:
        """
        删除预设问题模板

        Args:
            template_id: 模板ID
        """
        conn = self._connect()
        try:
            conn.execute("DELETE FROM kb_qa_templates WHERE id = ?", (template_id,))
            conn.commit()
        finally:
            conn.close()

    def get_conversation_summary(self, session_id: str) -> str:
        """
        生成对话历史摘要（用于注入到LLM上下文）

        Args:
            session_id: 会话标识

        Returns:
            对话摘要文本
        """
        convs = self.get_conversations(session_id, limit=10)
        if not convs:
            return "无历史对话"

        summary_lines = []
        for conv in convs[-6:]:  # 只取最近6轮
            role_name = "用户" if conv["role"] == "user" else "助手"
            content = conv["content"][:100]  # 每条最多100字
            summary_lines.append(f"- {role_name}: {content}...")

        return "\n".join(summary_lines)

    def export_conversation(
        self, session_id: str, format: str = "markdown"
    ) -> str:
        """
        导出对话历史

        Args:
            session_id: 会话标识
            format: 导出格式（markdown/pdf，目前只支持markdown）

        Returns:
            导出的文本内容
        """
        convs = self.get_conversations(session_id)

        if format == "markdown":
            lines = [
                f"# 科研问答记录",
                f"",
                f"**会话ID**: {session_id}",
                f"**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**对话条数**: {len(convs)}",
                f"",
                f"---",
                f"",
            ]
            for conv in convs:
                role_name = "👤 用户" if conv["role"] == "user" else "🤖 助手"
                lines.append(f"## {role_name}")
                lines.append(conv["content"])

                # 显示来源
                if conv["sources"]:
                    sources = json.loads(conv["sources"])
                    lines.append("")
                    lines.append("**来源引用**:")
                    for i, src in enumerate(sources[:3], 1):
                        title = src.get("title", "未知来源")
                        score = src.get("score", 0)
                        lines.append(f"{i}. {title} (相似度: {score:.2f})")

                lines.append("")
                lines.append("---")
                lines.append("")

            return "\n".join(lines)

        raise ValueError(f"不支持的格式: {format}")

    def get_session_list(self) -> List[Dict]:
        """
        获取所有会话列表（用于侧边栏显示）

        Returns:
            会话列表，包含会话ID和最后一条消息
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT
                    session_id,
                    MAX(created_at) as last_activity,
                    COUNT(*) as message_count,
                    (SELECT content FROM kb_conversations c2
                     WHERE c2.session_id = kb_conversations.session_id
                     ORDER BY created_at DESC LIMIT 1) as last_message
                FROM kb_conversations
                GROUP BY session_id
                ORDER BY last_activity DESC
                LIMIT 20
                """
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
