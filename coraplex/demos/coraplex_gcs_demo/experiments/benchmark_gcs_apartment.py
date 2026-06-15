"""
End-to-end performance benchmark for the Graph of Convex Sets free-space pipeline
on the IAI Apartment world loaded via the semantic_digital_twin package.

The benchmark measures each phase of the pipeline independently to show exactly
where time is spent:

  Phase 1 – World loading    : URDF parse and pybullet collision setup
  Phase 2 – Search space     : define the bounding box of the navigable volume
  Phase 3 – Collect obstacles: gather all obstacle bounding boxes from the world
  Phase 4+5 – Free space     : bounded incremental subtraction via subtract_disjoint
  Phase 6 – Materialise      : convert the free-space Event to a BoundingBoxCollection
  Phase 7 – Connectivity     : build the R-tree intersection graph

Run with:
    python3 coraplex/demos/coraplex_gcs_demo/experiments/benchmark_gcs_apartment.py

Requirements (all installed in-repo):
    semantic_digital_twin, random_events, rtree, trimesh, urdf_parser_py
"""
from __future__ import annotations

import time
import statistics

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    GraphOfConvexSets,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticEnvironmentAnnotation,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

import os

_APARTMENT_URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "semantic_digital_twin", "resources", "urdf", "apartment.urdf",
)


def _format_milliseconds(seconds: float) -> str:
    """Format a duration given in seconds as a human-readable milliseconds string."""
    return f"{seconds * 1000:.1f} ms"


def _time_function(label: str, function_to_time, *, repetitions: int = 1):
    """
    Run function_to_time the requested number of times, print the mean (and
    standard deviation when repetitions > 1), and return the last result.
    """
    elapsed_times = []
    result = None
    for _ in range(repetitions):
        start_time = time.perf_counter()
        result = function_to_time()
        elapsed_times.append(time.perf_counter() - start_time)
    mean_time = statistics.mean(elapsed_times)
    if repetitions > 1:
        standard_deviation = statistics.stdev(elapsed_times)
        print(
            f"  {label:<40s}  {_format_milliseconds(mean_time):>10s}"
            f"  ±{_format_milliseconds(standard_deviation)}"
        )
    else:
        print(f"  {label:<40s}  {_format_milliseconds(mean_time):>10s}")
    return result


def main():
    print("=" * 65)
    print("Graph of Convex Sets Free-Space Benchmark  –  IAI Apartment World")
    print("=" * 65)

    # ── Phase 1: World loading ───────────────────────────────────────────
    print("\n[1] World loading")

    def _load_apartment_world():
        parser = URDFParser.from_file(_APARTMENT_URDF_PATH)
        return parser.parse()

    world = _time_function("URDF parse + pybullet setup", _load_apartment_world)

    body_count = len(list(world.bodies))
    collision_body_count = sum(
        1 for body in world.bodies if isinstance(body, Body) and body.has_collision()
    )
    print(f"      bodies={body_count}  with-collision={collision_body_count}")

    # ── Phase 2: Define search space ─────────────────────────────────────
    print("\n[2] Search space")
    # The apartment furniture root is at (8.85, 1.75, 0).
    # Walls span roughly x ∈ [-1, 12],  y ∈ [-3, 5],  z ∈ [0, 3].
    search_space = BoundingBoxCollection(
        shapes=[
            BoundingBox(
                min_x=-1.0,
                min_y=-3.0,
                min_z=0.0,
                max_x=12.0,
                max_y=5.0,
                max_z=3.0,
                origin=HomogeneousTransformationMatrix(reference_frame=world.root),
            )
        ],
        reference_frame=world.root,
    )
    search_event = search_space.event
    print(f"  Search box  x[-1,12]  y[-3,5]  z[0,3]")
    print(f"  Search event variables: {list(search_event.variables)}")

    # ── Phase 3: Collect obstacle bounding boxes ─────────────────────────
    print("\n[3] Collect obstacle bounding boxes")

    def _collect_obstacle_bounding_boxes():
        annotation = SemanticEnvironmentAnnotation(root=world.root, _world=world)
        origin = HomogeneousTransformationMatrix(reference_frame=world.root)
        return list(annotation.as_bounding_box_collection_at_origin(origin))

    obstacle_bounding_boxes = _time_function(
        "as_bounding_box_collection_at_origin", _collect_obstacle_bounding_boxes
    )
    print(f"      obstacle bounding boxes: {len(obstacle_bounding_boxes)}")

    # ── Phases 4+5: Free space via subtract_disjoint ──────────────────────
    print("\n[4+5] Free-space via subtract_disjoint  [bounded incremental subtraction]")

    def _compute_free_space():
        free_space_accumulator = search_event
        for bounding_box in obstacle_bounding_boxes:
            obstacle = bounding_box.simple_event.as_composite_set() & search_event
            if not obstacle.is_empty():
                free_space_accumulator = free_space_accumulator.subtract_disjoint(obstacle)
        return free_space_accumulator

    free_space = _time_function(
        "subtract_disjoint loop (all obstacles)", _compute_free_space, repetitions=3
    )
    print(f"      free-space simple sets: {len(list(free_space.simple_sets))}")

    # ── Phase 6: Materialise into BoundingBoxCollection ──────────────────
    print("\n[6] Materialise free space into BoundingBoxCollection")

    def _materialise_free_space():
        return BoundingBoxCollection.from_event(
            reference_frame=world.root, event=free_space
        )

    free_space_collection = _time_function(
        "BoundingBoxCollection.from_event", _materialise_free_space, repetitions=3
    )
    print(f"      free-space bounding boxes: {len(free_space_collection)}")

    # ── Phase 7: Connectivity (R-tree) ────────────────────────────────────
    print("\n[7] Connectivity (R-tree)")

    def _compute_connectivity():
        graph_of_convex_sets = GraphOfConvexSets(world=world, search_space=search_space)
        for bounding_box in free_space_collection:
            graph_of_convex_sets.add_node(bounding_box)
        graph_of_convex_sets.calculate_connectivity(tolerance=0.001)
        return graph_of_convex_sets

    connectivity_graph = _time_function(
        "calculate_connectivity", _compute_connectivity, repetitions=3
    )
    print(
        f"      nodes={len(connectivity_graph.graph.nodes())}"
        f"  edges={len(connectivity_graph.graph.edges())}"
    )

    # ── Phase 8: End-to-end (single call) ────────────────────────────────
    print("\n[8] Full end-to-end: GraphOfConvexSets.free_space_from_world")

    def _run_end_to_end():
        loaded_world = _load_apartment_world()
        apartment_search_space = BoundingBoxCollection(
            shapes=[
                BoundingBox(
                    min_x=-1.0,
                    min_y=-3.0,
                    min_z=0.0,
                    max_x=12.0,
                    max_y=5.0,
                    max_z=3.0,
                    origin=HomogeneousTransformationMatrix(reference_frame=loaded_world.root),
                )
            ],
            reference_frame=loaded_world.root,
        )
        return GraphOfConvexSets.free_space_from_world(loaded_world, apartment_search_space)

    end_to_end_graph = _time_function(
        "free_space_from_world (incl. world load)", _run_end_to_end
    )
    print(
        f"      nodes={len(end_to_end_graph.graph.nodes())}"
        f"  edges={len(end_to_end_graph.graph.edges())}"
    )

    print("\n" + "=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
