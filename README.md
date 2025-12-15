
---

# Gemini XML Translator for Age of Empires III (Wars of Liberty)

Script en **Python** para traducir archivos XML del mod **Age of Empires III: Wars of Liberty** utilizando **Google Gemini (gemini-2.5-flash)**, optimizado con procesamiento multihilo, protección de tokens y caché automática.

Diseñado específicamente para localización de videojuegos históricos (1789–1916) con reglas estrictas de traducción para **español latinoamericano**.

---

## Requisitos

Antes de ejecutar el script, asegúrate de tener lo siguiente:

* Python 3.10 o superior instalado
* Acceso a CMD o PowerShell
* Una API Key válida de Google Gemini

---

## Comandos a ejecutar en CMD / PowerShell

### 1. Verificar que Python esté instalado

```bat
python --version
```

Si el comando no existe, instala Python desde [https://www.python.org](https://www.python.org) y marca la opción **“Add Python to PATH”**.

---

### 2. (Opcional) Crear y activar un entorno virtual

```bat
python -m venv venv
venv\Scripts\activate
```

---

### 3. Instalar dependencias necesarias

```bat
pip install google-generativeai tqdm
```

---

### 4. Ubicarse en la carpeta del script

Ejemplo:

```bat
cd C:\Users\User\Documents\translator
```

Asegúrate de que el archivo `translate_gemini.py` esté en esta carpeta.

---

### 5. Ejecutar el script

Ejemplo completo de ejecución:

```bat
python translate_gemini.py "unithelpstringsy.xml" "unithelpstringsy_es_latam.xml" --api-key "TU_API_KEY_AQUI" --source "English" --target "Latin American Spanish"
```

---

## Parámetros del comando

| Parámetro                       | Descripción                                          |
| ------------------------------- | ---------------------------------------------------- |
| `unithelpstringsy.xml`          | Archivo XML original (inglés)                        |
| `unithelpstringsy_es_latam.xml` | Archivo XML traducido                                |
| `--api-key`                     | Obligatorio. API Key de Google Gemini                |
| `--source`                      | Idioma origen (por defecto: English)                 |
| `--target`                      | Idioma destino (por defecto: Latin American Spanish) |

---

## Caché y reanudación

* Se genera automáticamente:

  ```text
  unithelpstringsy_es_latam.xml.cache.json
  ```
* Si el proceso se interrumpe, vuelve a ejecutar **el mismo comando** y el script continuará desde donde se quedó.

---

## Reglas importantes

* No se modifican tokens, placeholders ni saltos de línea
* No se agregan explicaciones ni texto adicional
* Se mantiene el mismo número y orden de strings
* Español neutral latinoamericano
* Uso de “Ustedes”, nunca “Vosotros”

---

## Uso recomendado

* Archivos XML grandes
* Proyectos de localización de videojuegos
* Mods de Age of Empires III
* Traducciones que requieren consistencia histórica

---

## Licencia

Uso libre para proyectos personales y de modding.
Revisa las políticas de Google Gemini para uso comercial.



