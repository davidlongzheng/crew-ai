from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

# class SupabaseConfig(BaseModel):
#     """Backend database configuration parameters.
#
#     Attributes:
#         dsn:
#             DSN for target database.
#     """
#
#     url: str
#     service_role_key: str
#


class CORSConfig(BaseSettings):
    allow_origins: list[str] = [
        "http://localhost:3000",
        "https://crew-ai-gamma.vercel.app",
        "http://192.168.1.151:3000",
    ]  # Frontend URL
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]


class Config(BaseSettings):
    """API configuration parameters.

    Automatically read modifications to the configuration parameters
    from environment variables and ``.env`` file.

    Attributes:
        database:
            Database configuration settings.
            Instance of :class:`app.backend.config.DatabaseConfig`.
        token_key:
            Random secret key used to sign JWT tokens.
        cors:
            CORS configuration settings.
            Instance of :class:`app.backend.config.CORSConfig`.
    """

    cors: CORSConfig = CORSConfig()

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )


config = Config()  # type: ignore[call-arg]
