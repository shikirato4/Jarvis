# SearXNG opcional para JARVIS

JARVIS funciona sin SearXNG. Para añadir cobertura local con la imagen oficial:

```powershell
docker run --name jarvis-searxng -d -p 8080:8080 `
  -e SEARXNG_BASE_URL=http://localhost:8080/ `
  searxng/searxng:latest
```

Habilita `json` en `search.formats` dentro de la configuración de SearXNG y
define en el `.env` de JARVIS:

```dotenv
SEARXNG_BASE_URL=http://127.0.0.1:8080
```

Comprueba el estado sin revelar configuración sensible:

```powershell
python -m jarvis web status
python -m jarvis web health
```

Detener y volver a iniciar el servicio:

```powershell
docker stop jarvis-searxng
docker start jarvis-searxng
```

Para retirarlo de la máquina:

```powershell
docker rm -f jarvis-searxng
```
