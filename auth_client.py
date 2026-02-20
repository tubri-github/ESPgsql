# FastAPI项目SSO认证客户端（纯Cookie/Session认证）
# 通过转发 auth_session cookie 到认证中心验证用户身份

import httpx
from typing import Optional, List
from fastapi import Request, HTTPException, Depends, status
from pydantic import BaseModel
import secrets
import urllib.parse


# 配置类
class AuthConfig:
    def __init__(
        self,
        auth_center_url: str = "http://localhost:8000",
        project_code: str = "project-a",
        api_prefix: str = "/api/v1",
        current_domain: str = "http://localhost:8001",
    ):
        self.auth_center_url = auth_center_url.rstrip('/')
        self.project_code = project_code
        self.api_prefix = api_prefix
        self.current_domain = current_domain.rstrip('/')


# 用户信息模型
class UserInfo(BaseModel):
    user_id: str
    username: str
    email: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    is_active: bool
    is_verified: bool
    is_superuser: bool
    project: str
    permissions: List[str] = []
    project_access: bool = True


# SSO认证客户端
class SSOAuthClient:
    def __init__(self, config: AuthConfig):
        self.config = config

    def get_login_url(self, redirect_uri: str, state: Optional[str] = None) -> str:
        """生成登录URL"""
        if not state:
            state = secrets.token_urlsafe(32)

        params = {
            "redirect_uri": redirect_uri,
            "project": self.config.project_code
        }
        if state:
            params["state"] = state

        query_string = urllib.parse.urlencode(params)
        return f"{self.config.auth_center_url}{self.config.api_prefix}/sso/login?{query_string}"

    async def check_session(self, request: Request) -> Optional[UserInfo]:
        """通过 auth_session cookie 检查会话状态"""
        try:
            auth_session = request.cookies.get("auth_session")
            if not auth_session:
                return None

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.config.auth_center_url}{self.config.api_prefix}/sso/check-session",
                    params={"project": self.config.project_code},
                    cookies={"auth_session": auth_session},
                    timeout=30.0
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success") and result["data"].get("has_project_access"):
                        data = result["data"]
                        return UserInfo(
                            user_id=data["user_id"],
                            username=data["username"],
                            email=data["email"],
                            display_name=data.get("display_name"),
                            is_active=True,
                            is_verified=True,
                            is_superuser=data.get("is_superuser", False),
                            project=self.config.project_code,
                            permissions=data.get("permissions", []),
                            project_access=True
                        )
                return None
        except Exception:
            return None

    async def logout(self, redirect_uri: Optional[str] = None) -> str:
        """生成登出URL"""
        params = {}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri

        query_string = urllib.parse.urlencode(params) if params else ""
        logout_url = f"{self.config.auth_center_url}{self.config.api_prefix}/sso/logout"

        if query_string:
            logout_url += f"?{query_string}"

        return logout_url


# 全局认证客户端实例
_auth_client: Optional[SSOAuthClient] = None
_auth_config: Optional[AuthConfig] = None


def init_auth(config: AuthConfig):
    """初始化认证配置"""
    global _auth_client, _auth_config
    _auth_config = config
    _auth_client = SSOAuthClient(config)


def get_auth_client() -> SSOAuthClient:
    """获取认证客户端"""
    if not _auth_client:
        raise RuntimeError("认证客户端未初始化，请先调用 init_auth()")
    return _auth_client


def get_auth_config() -> AuthConfig:
    """获取认证配置"""
    if not _auth_config:
        raise RuntimeError("认证配置未初始化，请先调用 init_auth()")
    return _auth_config


# 依赖函数：获取当前用户（可选）
async def get_current_user_optional(request: Request) -> Optional[UserInfo]:
    """获取当前用户（可选，未登录返回None）"""
    auth_client = get_auth_client()
    return await auth_client.check_session(request)


# 依赖函数：获取当前用户（必须）
async def get_current_user(
    request: Request,
    user: Optional[UserInfo] = Depends(get_current_user_optional)
) -> UserInfo:
    """获取当前用户（必须登录）"""
    if not user:
        current_url = str(request.url)
        auth_client = get_auth_client()
        login_url = auth_client.get_login_url(current_url)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要登录",
            headers={"Location": login_url}
        )
    return user


# 权限检查依赖
def require_permissions(*permissions: str):
    """要求特定权限的依赖函数"""
    async def check_permissions(user: UserInfo = Depends(get_current_user)):
        if not user.is_superuser:
            missing_permissions = [p for p in permissions if p not in user.permissions]
            if missing_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"缺少权限: {', '.join(missing_permissions)}"
                )
        return user
    return check_permissions