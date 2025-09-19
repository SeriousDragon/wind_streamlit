import os, tempfile, requests
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics import YOLO

# # ---------- НАСТРОЙКИ СТРАНИЦЫ ----------
# st.set_page_config(page_title="Wind Turbines vs Cable Towers — My Model", layout="wide")
# st.title("⚡ Wind Turbine Detector")

# st.write("Загрузи несколько файлов или укажи ссылку — модель сделает предикт. ")

# # ---------- Сайдбар: загрузка весов и метаданных ----------
# with st.sidebar:
#     st.header("⚙️ Модель и параметры")

#     weights_path = st.text_input("Путь к весам .pt", value="best.pt")
#     imgsz = st.number_input("imgsz", value=960, step=64, min_value=256, max_value=2048)
#     iou = st.slider("NMS IoU", 0.30, 0.90, 0.65, 0.01)
#     base_conf = st.slider("Базовый conf (собрать максимум кандидатов)", 0.00, 0.50, 0.05, 0.01)
#     device = st.text_input("device", value="0")

#     st.divider()
#     st.subheader("Классы для отображения")

#     # пробуем прочитать имена классов прямо из весов
#     try:
#         _tmp = YOLO(weights_path)
#         names = _tmp.model.names
#     except Exception:
#         names = {0: "class0", 1: "class1"}  # запасной вариант

#     selected_classes = st.multiselect(
#     "Какие классы показывать:",
#     options=list(names.values()),
#     default=[names[1]] if 1 in names else list(names.values())[:1]
#     )
#     allowed_ids = [cid for cid, cname in names.items() if cname in selected_classes]


# # ---------- утилиты ----------
# @st.cache_resource(show_spinner=False)
# def load_model(weights):
#     return YOLO(weights)


# def apply_class_thresholds(result, class_conf, allowed_classes=None):
#     """Фильтрует боксы: оставляет только выбранные классы и только выше своего порога."""
#     if result.boxes is None or len(result.boxes) == 0:
#         return result
#     keep_idx = []
#     for i, (cls_t, score_t) in enumerate(zip(result.boxes.cls, result.boxes.conf)):
#         cid = int(cls_t.item())
#         if allowed_classes is not None and cid not in allowed_classes:
#             continue
#         thr = class_conf.get(cid, 0.25)
#         if float(score_t) >= thr:
#             keep_idx.append(i)
#     result.boxes = result.boxes[keep_idx]
#     return result


# def plot_result(result, title=""):
#     im = result.plot()[:, :, ::-1]  # BGR->RGB
#     fig = plt.figure(figsize=(8,6))
#     plt.imshow(im); plt.axis("off"); plt.title(title)
#     return fig

# def save_upload_to_temp(uploaded_file):
#     suffix = Path(uploaded_file.name).suffix
#     tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
#     tf.write(uploaded_file.read()); tf.flush(); tf.close()
#     return tf.name

# # ---------- загрузка модели ----------
# try:
#     model = load_model(weights_path)
# except Exception as e:
#     st.error(f"Не удалось загрузить веса: {e}")
#     st.stop()

# # ---------- классы и пороги ----------
# names_from_weights = model.model.names
# names = names_from_weights

# with st.sidebar:
#     st.subheader("Пороги на классы")
#     # возьмём первые два класса или те, что из meta
#     cls_ids = sorted(list(names.keys())) if isinstance(names, dict) else [0,1]
#     # дефолты
#     default_map = {0: 0.35, 1: 0.15}
#     class_conf = {}
#     for cid in cls_ids:
#         cname = names[cid] if isinstance(names, dict) else str(cid)
#         default = default_map.get(cid, 0.25)
#         class_conf[cid] = st.slider(f"{cname}", 0.0, 1.0, float(default), 0.01)

# # ---------- Загрузка файлов и/или URL ----------
# import os, tempfile, requests, mimetypes, hashlib
# from pathlib import Path

# # --- persistent state ---
# if "sources" not in st.session_state:
#     st.session_state.sources = []           # пути к временным файлам
# if "seen_upload_ids" not in st.session_state:
#     st.session_state.seen_upload_ids = set()  # уникальные id загруженных файлов (из uploader)
# if "seen_url_ids" not in st.session_state:
#     st.session_state.seen_url_ids = set()     # уникальные id скачанных по URL

# def _file_uid_from_bytes(name: str, data: bytes) -> str:
#     # uid = имя + md5 содержимого (надёжная дедупликация)
#     h = hashlib.md5(data).hexdigest()
#     return f"{name}:{h}"

# def _save_bytes_to_temp(name: str, data: bytes) -> str:
#     suffix = Path(name).suffix or ".bin"
#     tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
#     tf.write(data); tf.flush(); tf.close()
#     return tf.name

# def _download_url_to_temp(url: str, timeout=40) -> tuple[str, str]:
#     r = requests.get(url, stream=True, timeout=timeout)
#     r.raise_for_status()
#     # определить расширение
#     suffix = Path(url).suffix
#     if not suffix or len(suffix) > 6:
#         ctype = r.headers.get("content-type", "").split(";")[0].strip()
#         suffix = mimetypes.guess_extension(ctype) or ".jpg"
#     # собрать байты (и заодно md5)
#     md5 = hashlib.md5()
#     tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
#     with open(tf.name, "wb") as f:
#         for chunk in r.iter_content(chunk_size=8192):
#             if chunk:
#                 md5.update(chunk)
#                 f.write(chunk)
#     uid = f"{url}:{md5.hexdigest()}"
#     return tf.name, uid

# st.subheader("📤 Загрузка данных")

# col1, col2, col3 = st.columns([2,1,1])

# with col1:
#     up_files = st.file_uploader(
#         "Кинь сюда несколько изображений/видео",
#         type=["jpg","jpeg","png","bmp","webp","mp4","avi","mov","mkv"],
#         accept_multiple_files=True,
#         key="uploader_files"
#     )
#     added = []
#     if up_files:
#         for f in up_files:
#             try:
#                 data = f.getvalue()  # читаем байты (не «сгорает» при повторном доступе)
#                 uid = _file_uid_from_bytes(f.name, data)
#                 if uid in st.session_state.seen_upload_ids:
#                     continue  # уже добавляли
#                 p = _save_bytes_to_temp(f.name, data)
#                 if os.path.exists(p):
#                     st.session_state.sources.append(p)
#                     st.session_state.seen_upload_ids.add(uid)
#                     added.append(Path(p).name)
#             except Exception as e:
#                 st.error(f"Не удалось сохранить {f.name}: {e}")
#     if added:
#         st.success(f"Добавлено уникальных файлов: {len(added)}")

# with col2:
#     url_input = st.text_input("Прямая ссылка на файл", key="url_input")
#     if st.button("Скачать по ссылке", key="btn_download_url"):
#         if not url_input.strip():
#             st.warning("Вставьте ссылку перед скачиванием.")
#         else:
#             try:
#                 with st.spinner("Скачиваю…"):
#                     p, uid = _download_url_to_temp(url_input.strip(), timeout=40)
#                 if uid in st.session_state.seen_url_ids:
#                     # уже скачивали ровно этот контент
#                     os.remove(p)  # уберём дубликат
#                     st.info("Этот файл уже был добавлен (дубликат по содержимому).")
#                 else:
#                     st.session_state.seen_url_ids.add(uid)
#                     st.session_state.sources.append(p)
#                     st.success(f"Скачано: {Path(p).name}")
#             except Exception as e:
#                 st.error(f"Не удалось скачать: {e}")

# with col3:
#     if st.button("🗑️ Очистить список", key="btn_clear_sources"):
#         # чистим всё, включая «виденные» id
#         st.session_state.sources.clear()
#         st.session_state.seen_upload_ids.clear()
#         st.session_state.seen_url_ids.clear()
#         st.success("Список источников очищен.")

# # отображение текущего списка
# sources = st.session_state.sources
# if sources:
#     st.caption("Файлы, готовые к детекции:")
#     st.write("\n".join(f"- {Path(p).name}" for p in sources))
# else:
#     st.info("Пока нет файлов. Загрузите их или скачайте по ссылке.")

# # ---------- Детекция ----------
# # ====== ДЕТЕКЦИЯ (вставь целиком этот блок) ======
# import numpy as np
# from pathlib import Path

# st.subheader("🔎 Детекция")

# # грузим модель один раз (без try, чтобы не было незакрытых блоков)
# model = load_model(weights_path)

# def apply_class_thresholds(result, class_conf, allowed_classes=None):
#     """Оставляем боксы только выбранных классов и только выше их порогов."""
#     if result.boxes is None or len(result.boxes) == 0:
#         return result
#     cls_np  = result.boxes.cls.cpu().numpy().astype(int)
#     conf_np = result.boxes.conf.cpu().numpy().astype(float)
#     thr_vec = np.array([float(class_conf.get(int(c), 0.25)) for c in cls_np], dtype=float)

#     mask = conf_np >= thr_vec
#     if allowed_classes is not None:
#         mask &= np.isin(cls_np, np.array(list(allowed_classes), dtype=int))

#     keep_idx = np.where(mask)[0].tolist()
#     result.boxes = result.boxes[keep_idx] if keep_idx else result.boxes[[]]
#     return result

# def run_one(src_path: str):
#     # Предикт по выбранным классам (allowed_ids может быть пустым → берём все)
#     results = model.predict(
#         source=src_path,
#         imgsz=int(imgsz),
#         conf=float(base_conf),            # глобально низкий, потом режем своими порогами
#         iou=float(iou),
#         device=device,
#         classes=allowed_ids if allowed_ids else None,
#         verbose=False
#     )
#     r = results[0]
#     # Пост-фильтр ДО отрисовки
#     r = apply_class_thresholds(r, class_conf, allowed_classes=set(allowed_ids) if allowed_ids else None)
#     # Рисуем
#     st.pyplot(plot_result(r, title=Path(src_path).name))

# run_det = st.button("Запустить детекцию", disabled=(len(sources) == 0))
# if run_det:
#     if len(sources) == 1:
#         run_one(sources[0])
#     else:
#         labels = [Path(s).name for s in sources]
#         tabs = st.tabs(labels)  # создаём по вкладке на каждый файл
#         for tab, src in zip(tabs, sources):
#             with tab:
#                 run_one(src)


# ==================== ВКЛАДКИ СТРАНИЦЫ ====================
tab_detect, tab_report = st.tabs(["🔎 Детекция", "📑 Отчёт об обучении"])

# ----- ТАБ: ДЕТЕКЦИЯ (перенеси сюда свой текущий код детекции) -----
with tab_detect:
# ---------- Сайдбар: загрузка весов и метаданных ----------
    with st.sidebar:
        st.header("⚙️ Модель и параметры")

        weights_path = st.text_input("Путь к весам .pt", value="best.pt")
        imgsz = st.number_input("imgsz", value=960, step=64, min_value=256, max_value=2048)
        iou = st.slider("NMS IoU", 0.30, 0.90, 0.65, 0.01)
        base_conf = st.slider("Базовый conf (собрать максимум кандидатов)", 0.00, 0.50, 0.05, 0.01)
        device_choice = st.selectbox("Device", ["auto", "cpu", "cuda:0"], index=0)

        st.divider()
        st.subheader("Классы для отображения")

        # пробуем прочитать имена классов прямо из весов
        try:
            _tmp = YOLO(weights_path)
            names = _tmp.model.names
        except Exception:
            names = {0: "class0", 1: "class1"}  # запасной вариант

        selected_classes = st.multiselect(
        "Какие классы показывать:",
        options=list(names.values()),
        default=[names[1]] if 1 in names else list(names.values())[:1]
        )
        allowed_ids = [cid for cid, cname in names.items() if cname in selected_classes]


    # ---------- утилиты ----------
    @st.cache_resource(show_spinner=False)
    def load_model(weights):
        return YOLO(weights)
    
    def resolve_device(choice: str):
        choice = (choice or "").strip().lower()
        if choice in ("", "auto"):
            return 0 if torch.cuda.is_available() else "cpu"
        if choice in ("cpu", "-1"):
            return "cpu"
        if choice.startswith("cuda") or choice.isdigit():
            # если GPU не доступен — тихо уходим на CPU
            return 0 if torch.cuda.is_available() else "cpu"
        return "cpu"



    def apply_class_thresholds(result, class_conf, allowed_classes=None):
        """Фильтрует боксы: оставляет только выбранные классы и только выше своего порога."""
        if result.boxes is None or len(result.boxes) == 0:
            return result
        keep_idx = []
        for i, (cls_t, score_t) in enumerate(zip(result.boxes.cls, result.boxes.conf)):
            cid = int(cls_t.item())
            if allowed_classes is not None and cid not in allowed_classes:
                continue
            thr = class_conf.get(cid, 0.25)
            if float(score_t) >= thr:
                keep_idx.append(i)
        result.boxes = result.boxes[keep_idx]
        return result


    def plot_result(result, title=""):
        im = result.plot()[:, :, ::-1]  # BGR->RGB
        fig = plt.figure(figsize=(8,6))
        plt.imshow(im); plt.axis("off"); plt.title(title)
        return fig

    def save_upload_to_temp(uploaded_file):
        suffix = Path(uploaded_file.name).suffix
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tf.write(uploaded_file.read()); tf.flush(); tf.close()
        return tf.name

    # ---------- загрузка модели ----------
    try:
        model = load_model(weights_path)
    except Exception as e:
        st.error(f"Не удалось загрузить веса: {e}")
        st.stop()

    # ---------- классы и пороги ----------
    names_from_weights = model.model.names
    names = names_from_weights

    with st.sidebar:
        st.subheader("Пороги на классы")
        # возьмём первые два класса или те, что из meta
        cls_ids = sorted(list(names.keys())) if isinstance(names, dict) else [0,1]
        # дефолты
        default_map = {0: 0.35, 1: 0.15}
        class_conf = {}
        for cid in cls_ids:
            cname = names[cid] if isinstance(names, dict) else str(cid)
            default = default_map.get(cid, 0.25)
            class_conf[cid] = st.slider(f"{cname}", 0.0, 1.0, float(default), 0.01)

    # ---------- Загрузка файлов и/или URL ----------
    import os, tempfile, requests, mimetypes, hashlib
    from pathlib import Path

    # --- persistent state ---
    if "sources" not in st.session_state:
        st.session_state.sources = []           # пути к временным файлам
    if "seen_upload_ids" not in st.session_state:
        st.session_state.seen_upload_ids = set()  # уникальные id загруженных файлов (из uploader)
    if "seen_url_ids" not in st.session_state:
        st.session_state.seen_url_ids = set()     # уникальные id скачанных по URL

    def _file_uid_from_bytes(name: str, data: bytes) -> str:
        # uid = имя + md5 содержимого (надёжная дедупликация)
        h = hashlib.md5(data).hexdigest()
        return f"{name}:{h}"

    def _save_bytes_to_temp(name: str, data: bytes) -> str:
        suffix = Path(name).suffix or ".bin"
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tf.write(data); tf.flush(); tf.close()
        return tf.name

    def _download_url_to_temp(url: str, timeout=40) -> tuple[str, str]:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        # определить расширение
        suffix = Path(url).suffix
        if not suffix or len(suffix) > 6:
            ctype = r.headers.get("content-type", "").split(";")[0].strip()
            suffix = mimetypes.guess_extension(ctype) or ".jpg"
        # собрать байты (и заодно md5)
        md5 = hashlib.md5()
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with open(tf.name, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    md5.update(chunk)
                    f.write(chunk)
        uid = f"{url}:{md5.hexdigest()}"
        return tf.name, uid

    st.subheader("📤 Загрузка данных")

    col1, col2, col3 = st.columns([2,1,1])

    with col1:
        up_files = st.file_uploader(
            "Кинь сюда несколько изображений/видео",
            type=["jpg","jpeg","png","bmp","webp","mp4","avi","mov","mkv"],
            accept_multiple_files=True,
            key="uploader_files"
        )
        added = []
        if up_files:
            for f in up_files:
                try:
                    data = f.getvalue()  # читаем байты (не «сгорает» при повторном доступе)
                    uid = _file_uid_from_bytes(f.name, data)
                    if uid in st.session_state.seen_upload_ids:
                        continue  # уже добавляли
                    p = _save_bytes_to_temp(f.name, data)
                    if os.path.exists(p):
                        st.session_state.sources.append(p)
                        st.session_state.seen_upload_ids.add(uid)
                        added.append(Path(p).name)
                except Exception as e:
                    st.error(f"Не удалось сохранить {f.name}: {e}")
        if added:
            st.success(f"Добавлено уникальных файлов: {len(added)}")

    with col2:
        url_input = st.text_input("Прямая ссылка на файл", key="url_input")
        if st.button("Скачать по ссылке", key="btn_download_url"):
            if not url_input.strip():
                st.warning("Вставьте ссылку перед скачиванием.")
            else:
                try:
                    with st.spinner("Скачиваю…"):
                        p, uid = _download_url_to_temp(url_input.strip(), timeout=40)
                    if uid in st.session_state.seen_url_ids:
                        # уже скачивали ровно этот контент
                        os.remove(p)  # уберём дубликат
                        st.info("Этот файл уже был добавлен (дубликат по содержимому).")
                    else:
                        st.session_state.seen_url_ids.add(uid)
                        st.session_state.sources.append(p)
                        st.success(f"Скачано: {Path(p).name}")
                except Exception as e:
                    st.error(f"Не удалось скачать: {e}")

    with col3:
        if st.button("🗑️ Очистить список", key="btn_clear_sources"):
            # чистим всё, включая «виденные» id
            st.session_state.sources.clear()
            st.session_state.seen_upload_ids.clear()
            st.session_state.seen_url_ids.clear()
            st.success("Список источников очищен.")

    # отображение текущего списка
    sources = st.session_state.sources
    if sources:
        st.caption("Файлы, готовые к детекции:")
        st.write("\n".join(f"- {Path(p).name}" for p in sources))
    else:
        st.info("Пока нет файлов. Загрузите их или скачайте по ссылке.")

    # ---------- Детекция ----------
    # ====== ДЕТЕКЦИЯ (вставь целиком этот блок) ======
    import numpy as np
    from pathlib import Path

    st.subheader("🔎 Детекция")

    # грузим модель один раз (без try, чтобы не было незакрытых блоков)
    model = load_model(weights_path)

    def apply_class_thresholds(result, class_conf, allowed_classes=None):
        """Оставляем боксы только выбранных классов и только выше их порогов."""
        if result.boxes is None or len(result.boxes) == 0:
            return result
        cls_np  = result.boxes.cls.cpu().numpy().astype(int)
        conf_np = result.boxes.conf.cpu().numpy().astype(float)
        thr_vec = np.array([float(class_conf.get(int(c), 0.25)) for c in cls_np], dtype=float)

        mask = conf_np >= thr_vec
        if allowed_classes is not None:
            mask &= np.isin(cls_np, np.array(list(allowed_classes), dtype=int))

        keep_idx = np.where(mask)[0].tolist()
        result.boxes = result.boxes[keep_idx] if keep_idx else result.boxes[[]]
        return result

    def run_one(src_path: str):
        # Предикт по выбранным классам (allowed_ids может быть пустым → берём все)
        dev = resolve_device(device_choice)
        results = model.predict(
            source=src_path,
            imgsz=int(imgsz),
            conf=float(base_conf),            # глобально низкий, потом режем своими порогами
            iou=float(iou),
            device=dev,
            classes=allowed_ids if allowed_ids else None,
            verbose=False
        )
        r = results[0]
        # Пост-фильтр ДО отрисовки
        r = apply_class_thresholds(r, class_conf, allowed_classes=set(allowed_ids) if allowed_ids else None)
        # Рисуем
        st.pyplot(plot_result(r, title=Path(src_path).name))

    run_det = st.button("Запустить детекцию", disabled=(len(sources) == 0))
    if run_det:
        if len(sources) == 1:
            run_one(sources[0])
        else:
            labels = [Path(s).name for s in sources]
            tabs = st.tabs(labels)  # создаём по вкладке на каждый файл
            for tab, src in zip(tabs, sources):
                with tab:
                    run_one(src)

# ----- ТАБ: ОТЧЁТ ОБ ОБУЧЕНИИ -----
with tab_report:
    st.subheader("Информация об обучении")

    st.set_page_config(page_title="Сравнение обучения", layout="wide")
    st.title("📑 Отчёт об обучении: До vs После обогащения датасета")

    # ==== Блок 1. Сводка чисел ====
    st.header("Сводка")

    cols = st.columns(2)
    with cols[0]:
        st.subheader("До обогащения")
        st.metric("Эпохи", 95, 20, 'off')
        st.metric("Train", 2643)
        st.metric("Val", 247)
        st.metric("Test", 130)

        st.markdown("**Метрики:**")
        st.metric("Precision", "0.76")
        st.metric("Recall", "0.50")
        st.metric("mAP@0.5", "0.62")
        st.metric("mAP@0.5:0.95", "0.39")

    with cols[1]:
        st.subheader("После обогащения")
        st.metric("Эпохи", 154, 20, 'off')
        st.metric("Train", 29073)
        st.metric("Val", 280)
        st.metric("Test", 130)

        st.markdown("**Метрики:**")
        st.metric("Precision", "0.87")
        st.metric("Recall", "0.76")
        st.metric("mAP@0.5", "0.78")
        st.metric("mAP@0.5:0.95", "0.55")

    st.header("📊 Разбиение классов")

    # данные примеры — подставишь свои числа
    class_dist_before = {
        "cable tower": 467,
        "turbine": 16260
    }
    class_dist_after = {
        "cable tower": 51984,
        "turbine": 171198
    }

    # собираем в один DataFrame для наглядного сравнения
    df_split = pd.DataFrame({
        "Класс": list(class_dist_before.keys()),
        "До обогащения": list(class_dist_before.values()),
        "После обогащения": [class_dist_after[k] for k in class_dist_before]
    })

    st.dataframe(df_split, use_container_width=True)   
    st.markdown("---")

    # ==== Блок 2. Графики ====
    st.header("Графики")

    # Укажи свои пути (положи файлы в репо, например в assets/)
    PR_BEFORE  = "pr_curve_before.png"
    PR_AFTER   = "pr_curve_after.png"
    CM_BEFORE  = "assets/cm_before.png"
    CM_AFTER   = "assets/cm_after.png"
    CURVE_BEFORE = "assets/learning_curves_before.png"
    CURVE_AFTER  = "assets/learning_curves_after.png"

    # Функция-помощник: показать картинку если есть, иначе — опционально аплоадер
    def show_img_or_upload(path: str, title: str, allow_upload_fallback=False, upload_key=None):
        p = Path(path)
        st.subheader(title)
        if p.exists():
            st.image(str(p), use_container_width=True)
        elif allow_upload_fallback:
            up = st.file_uploader(f"Загрузить {title}", type=["png","jpg","jpeg","webp"], key=upload_key)
            if up: st.image(up, use_container_width=True)
        else:
            st.info(f"Файл не найден: `{path}`")

    # Ряды «до/после»
    row1 = st.columns(2)
    with row1[0]:
        show_img_or_upload(PR_BEFORE, "PR-кривая — До", allow_upload_fallback=False)
    with row1[1]:
        show_img_or_upload(PR_AFTER,  "PR-кривая — После", allow_upload_fallback=False)




