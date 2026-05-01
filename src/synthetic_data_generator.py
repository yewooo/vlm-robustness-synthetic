"""Synthetic image and question generation for controlled VQA experiments."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


IMG_SIZE = 256
BACKGROUND_COLOR = "white"
SHAPES = ["square", "circle", "triangle"]
COLORS = {
    "red": (220, 20, 60),
    "blue": (65, 105, 225),
    "green": (34, 139, 34),
    "yellow": (255, 215, 0),
}
QUERY_TYPES = ["existence", "count", "spatial", "comparison"]


@dataclass
class SceneObject:
    obj_id: int
    shape: str
    color: str
    x: int
    y: int
    size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def draw_object(draw: ImageDraw.ImageDraw, obj: SceneObject) -> None:
    half = obj.size // 2
    color_rgb = COLORS[obj.color]
    if obj.shape == "square":
        draw.rectangle([obj.x - half, obj.y - half, obj.x + half, obj.y + half], fill=color_rgb, outline="black")
    elif obj.shape == "circle":
        draw.ellipse([obj.x - half, obj.y - half, obj.x + half, obj.y + half], fill=color_rgb, outline="black")
    elif obj.shape == "triangle":
        points = [(obj.x, obj.y - half), (obj.x - half, obj.y + half), (obj.x + half, obj.y + half)]
        draw.polygon(points, fill=color_rgb, outline="black")
    else:
        raise ValueError(f"Unknown shape: {obj.shape}")


def sample_position(size: int, margin: int = 20) -> tuple[int, int]:
    half = size // 2
    x = random.randint(margin + half, IMG_SIZE - margin - half)
    y = random.randint(margin + half, IMG_SIZE - margin - half)
    return x, y


def generate_random_object(obj_id: int) -> SceneObject:
    size = random.randint(26, 42)
    x, y = sample_position(size)
    return SceneObject(
        obj_id=obj_id,
        shape=random.choice(SHAPES),
        color=random.choice(list(COLORS.keys())),
        x=x,
        y=y,
        size=size,
    )


def generate_scene(num_objects: int | None = None) -> tuple[Image.Image, list[SceneObject]]:
    if num_objects is None:
        num_objects = random.randint(2, 5)
    objects = [generate_random_object(i) for i in range(num_objects)]
    image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    for obj in objects:
        draw_object(draw, obj)
    return image, objects


def count_matching_objects(objects: list[SceneObject], color: str, shape: str) -> int:
    return sum(obj.color == color and obj.shape == shape for obj in objects)


def generate_query(objects: list[SceneObject], forced_query_type: str | None = None) -> tuple[str, str, dict]:
    query_type = forced_query_type or random.choice(QUERY_TYPES)

    if query_type == "count":
        color = random.choice(list(COLORS.keys()))
        shape = random.choice(SHAPES)
        count = count_matching_objects(objects, color, shape)
        return (
            f"How many {color} {shape}s are present?",
            str(count),
            {"query_type": "count", "target_color": color, "target_shape": shape},
        )

    if query_type == "existence":
        color = random.choice(list(COLORS.keys()))
        shape = random.choice(SHAPES)
        exists = count_matching_objects(objects, color, shape) > 0
        return (
            f"Is there a {color} {shape}?",
            "yes" if exists else "no",
            {
                "query_type": "existence",
                "target_color": color,
                "target_shape": shape,
                "exists_in_scene": exists,
            },
        )

    if query_type == "comparison":
        color_1 = random.choice(list(COLORS.keys()))
        shape_1 = random.choice(SHAPES)
        color_2 = random.choice(list(COLORS.keys()))
        shape_2 = random.choice(SHAPES)
        count_1 = count_matching_objects(objects, color_1, shape_1)
        count_2 = count_matching_objects(objects, color_2, shape_2)
        return (
            f"Are there more {color_1} {shape_1}s than {color_2} {shape_2}s?",
            "yes" if count_1 > count_2 else "no",
            {
                "query_type": "comparison",
                "target_1": {"color": color_1, "shape": shape_1, "count": count_1},
                "target_2": {"color": color_2, "shape": shape_2, "count": count_2},
            },
        )

    if query_type == "spatial":
        obj_1, obj_2 = random.sample(objects, 2)
        relation = random.choice(["left of", "above"])
        if relation == "left of":
            answer = "yes" if obj_1.x < obj_2.x else "no"
        else:
            answer = "yes" if obj_1.y < obj_2.y else "no"
        return (
            f"Is the {obj_1.color} {obj_1.shape} {relation} the {obj_2.color} {obj_2.shape}?",
            answer,
            {
                "query_type": "spatial",
                "relation": relation,
                "object_1": obj_1.to_dict(),
                "object_2": obj_2.to_dict(),
            },
        )

    raise ValueError(f"Unknown query type: {query_type}")


def generate_sample(num_objects: int | None = None, forced_query_type: str | None = None) -> dict:
    image, objects = generate_scene(num_objects)
    query, answer, metadata = generate_query(objects, forced_query_type)
    return {
        "image": image,
        "objects": [obj.to_dict() for obj in objects],
        "query": query,
        "answer": answer,
        "metadata": metadata,
    }


def generate_dataset(n: int, seed: int = 42) -> list[dict]:
    random.seed(seed)
    dataset = []
    for idx in range(n):
        forced_query_type = QUERY_TYPES[idx % len(QUERY_TYPES)]
        dataset.append(generate_sample(forced_query_type=forced_query_type))
    random.shuffle(dataset)
    return dataset


def save_dataset(
    dataset: list[dict],
    dataset_name: str,
    output_dir: str | Path,
    image_dir: str | Path,
    add_boundary: bool = True,
) -> Path:
    """Save images and JSON records.

    The optional boundary makes white-background images easier to inspect in
    reports and slides.
    """

    output_dir = Path(output_dir)
    image_dir = Path(image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, sample in enumerate(dataset):
        filename = f"{dataset_name}_{idx:05d}.png"
        image = sample["image"]
        if add_boundary:
            image = ImageOps.expand(image, border=2, fill="#444444")
        image.save(image_dir / filename)
        record = {k: v for k, v in sample.items() if k != "image"}
        record["image_filename"] = filename
        records.append(record)

    json_path = output_dir / f"{dataset_name}.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return json_path
