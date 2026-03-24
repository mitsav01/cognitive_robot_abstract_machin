from __future__ import annotations

import datetime
import logging
from copy import deepcopy
from dataclasses import dataclass, field, InitVar
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas
import pandas as pd

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.adapters.qp_adapter import QPDataSymbolic
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.exceptions import (
    HardConstraintsViolatedException,
)
from giskardpy.qp.qp_data import (
    QPDataExplicit,
)
from giskardpy.qp.qp_data_factories import QPDataFactory
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.utils.utils import create_path
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller_config import QPControllerConfig

date_str = datetime.datetime.now().strftime("%Yy-%mm-%dd--%Hh-%Mm-%Ss")


@dataclass
class QPDebugger:
    qp_data_symbolic: QPDataSymbolic
    current_solution: np.ndarray | None = field(default=None)
    direct_limits: pandas.DataFrame = field(init=False)
    equality_constraints: pandas.DataFrame = field(init=False)
    equality_matrix: pandas.DataFrame = field(init=False)
    # weights: pandas.DataFrame = field(init=False)
    # A: pandas.DataFrame = field(init=False)
    # b: pandas.DataFrame = field(init=False)
    # E: pandas.DataFrame = field(init=False)
    # bE: pandas.DataFrame = field(init=False)
    # lbA: pandas.DataFrame = field(init=False)
    # ubA: pandas.DataFrame = field(init=False)
    # xdot: pandas.DataFrame = field(init=False)
    # debug: pandas.DataFrame = field(init=False)

    def __post_init__(self):
        self.update(self.current_solution)

    def update(self, current_solution: np.ndarray):
        self.current_solution = current_solution
        last_solution = (
            np.ones(self.qp_data_symbolic.box_lower_constraints.shape[0]) * np.nan
        )
        if self.current_solution is not None:
            last_solution[self.quadratic_weight_filter] = self.current_solution
        self.current_solution = last_solution
        self.create_direct_limits()
        self.create_equality_constraints()

    @property
    def quadratic_weight_filter(self) -> np.ndarray:
        quadratic_weight_filter = np.ones(
            self.qp_data_symbolic.quadratic_weights.shape[0]
        )
        quadratic_weight_filter[self.qp_data_symbolic.num_non_slack_variables :] = (
            self.qp_data_symbolic.quadratic_weights.evaluate()[
                self.qp_data_symbolic.num_non_slack_variables :
            ]
            != 0
        )
        return quadratic_weight_filter.astype(bool)

    def create_direct_limits(self):
        self.direct_limits = pd.DataFrame(
            {
                "lower bounds": self.qp_data_symbolic.box_lower_constraints.evaluate(),
                "solution": self.current_solution,
                "upper bounds": self.qp_data_symbolic.box_upper_constraints.evaluate(),
                "quadratic weight": self.qp_data_symbolic.quadratic_weights.evaluate(),
                "linear weight": self.qp_data_symbolic.linear_weights.evaluate(),
            },
            self.free_variable_names,
            dtype=float,
        )

    def create_equality_constraints(self):
        eq_matrix_dofs_np = self.qp_data_symbolic.eq_matrix_dofs.evaluate()
        eq_matrix_slack_np = self.qp_data_symbolic.eq_matrix_slack.evaluate()
        Ex = eq_matrix_dofs_np @ self.current_solution[: eq_matrix_dofs_np.shape[1]]
        bounds = self.qp_data_symbolic.eq_bounds.evaluate()
        self.equality_constraints = pd.DataFrame(
            {
                "Ex": Ex,
                "slack": bounds - Ex,
                "bounds": bounds,
            },
            self.equality_constr_names,
            dtype=float,
        )
        self.equality_matrix = pd.DataFrame(
            eq_matrix_dofs_np,
            self.equality_constr_names,
            self.degree_of_freedom_names,
            dtype=float,
        )

    def _has_nan(self):
        nan_entries = self.p_A[0].isnull().stack()
        row_col_names = nan_entries[nan_entries].index.tolist()
        pass

    def _print_pandas_array(self, array):
        import pandas as pd

        if len(array) > 0:
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(array)

    def save_pandas(
        self, dfs, names, path, time: float, folder_name: Optional[str] = None
    ):

        if folder_name is None:
            folder_name = ""
        folder_name = f"{path}/pandas/{folder_name}_{date_str}/{time}/"
        create_path(folder_name)
        for df, name in zip(dfs, names):
            csv_string = "name\n"
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                if df.shape[1] > 1:
                    for column_name, column in df.T.items():
                        zero_filtered_column = (
                            column.replace(0, np.nan)
                            .dropna(how="all")
                            .replace(np.nan, 0)
                        )
                        csv_string += zero_filtered_column.add_prefix(
                            column_name + "||"
                        ).to_csv(float_format="%.6f")
                else:
                    csv_string += df.to_csv(float_format="%.6f")
            file_name2 = f"{folder_name}{name}.csv"
            with open(file_name2, "w") as f:
                f.write(csv_string)

    def _update_quadratic_weights(self, qp_data: QPDataExplicit):
        self.p_weights = pd.DataFrame(
            {
                "quadratic": qp_data.quadratic_weights,
                "linear": qp_data.linear_weights,
            },
            self.free_variable_names,
            dtype=float,
        )

    def _update_linear_weights(self, qp_data: QPDataExplicit):
        self.p_g = pd.DataFrame(
            qp_data.linear_weights,
            self.free_variable_names,
            ["quadratic"],
            dtype=float,
        )

    def _update_box_constraints(self, qp_data: QPDataExplicit):
        self.p_lb = pd.DataFrame(
            qp_data.box_lower_constraints,
            self.free_variable_names,
            ["data"],
            dtype=float,
        )
        self.p_ub = pd.DataFrame(
            qp_data.box_upper_constraints,
            self.free_variable_names,
            ["data"],
            dtype=float,
        )
        self.p_b = pd.DataFrame(
            {
                "lb": qp_data.box_lower_constraints,
                "ub": qp_data.box_upper_constraints,
            },
            self.free_variable_names,
            dtype=float,
        )

    def _update_eq_constraints(self, qp_data: QPDataExplicit):
        if len(qp_data.equality_bounds) > 0:
            self.p_bE_raw = pd.DataFrame(
                qp_data.equality_bounds,
                self.equality_constr_names,
                ["data"],
                dtype=float,
            )
            self.p_bE = deepcopy(self.p_bE_raw)
            self.p_bE[
                len(
                    self.qp_controller.qp_data_factory.qp_data._equality_bounds.names_derivative_links
                ) :
            ] /= self.qp_controller.config.mpc_dt
        else:
            self.p_bE = pd.DataFrame()

    def _update_inequality_constraints(self, qp_data: QPDataExplicit):
        if len(qp_data.inequality_lower_bounds) > 0:
            self.p_lbA_raw = pd.DataFrame(
                qp_data.inequality_lower_bounds,
                self.inequality_constr_names,
                ["data"],
                dtype=float,
            )
            self.p_lbA = deepcopy(self.p_lbA_raw)
            self.p_lbA /= self.qp_controller.config.mpc_dt

            self.p_ubA_raw = pd.DataFrame(
                qp_data.inequality_upper_bounds,
                self.inequality_constr_names,
                ["data"],
                dtype=float,
            )
            self.p_ubA = deepcopy(self.p_ubA_raw)
            self.p_ubA /= self.qp_controller.config.mpc_dt

            self.p_bA_raw = pd.DataFrame(
                {
                    "lbA": qp_data.inequality_lower_bounds,
                    "ubA": qp_data.inequality_upper_bounds,
                },
                self.inequality_constr_names,
                dtype=float,
            )
            self.p_bA = deepcopy(self.p_bA_raw)
            self.p_bA /= self.qp_controller.config.mpc_dt
        else:
            self.p_lbA = pd.DataFrame()
            self.p_ubA = pd.DataFrame()

    def _update_equality_matrix(self, qp_data: QPDataExplicit):
        if len(qp_data.dense_eq_matrix) > 0:
            self.p_E = pd.DataFrame(
                qp_data.dense_eq_matrix,
                self.equality_constr_names,
                self.free_variable_names,
                dtype=float,
            )
        else:
            self.p_E = pd.DataFrame()

    def _update_inequality_matrix(self, qp_data: QPDataExplicit):
        if len(qp_data.dense_neq_matrix) > 0:
            self.p_A = pd.DataFrame(
                qp_data.dense_neq_matrix,
                self.inequality_constr_names,
                self.free_variable_names,
                dtype=float,
            )
        else:
            self.p_A = pd.DataFrame()

    def _update_xdot(
        self,
        qp_data: QPDataExplicit,
        new_xdot_full: Optional[np.ndarray],
    ):
        zero_quadratic_weight_filter = qp_data.quadratic_weights != 0
        zero_quadratic_weight_filter[: -qp_data.num_slack_variables] = True
        self.p_xdot = None
        if new_xdot_full is None:
            return

        num_vel_constr = len(
            self.qp_controller.constraint_collection.derivative_inequality_constraints
        ) * (self.qp_controller.config.prediction_horizon - 2)
        num_eq_vel_constr = len(
            self.qp_controller.constraint_collection.derivative_equality_constraints
        ) * (self.qp_controller.config.prediction_horizon - 2)
        num_neq_constr = len(
            self.qp_controller.constraint_collection.inequality_constraints
        )
        num_eq_constr = len(
            self.qp_controller.constraint_collection.equality_constraints
        )
        num_constr = num_vel_constr + num_neq_constr + num_eq_constr + num_eq_vel_constr

        xdot_full = np.ones(qp_data.quadratic_weights.shape) * np.nan
        xdot_full[zero_quadratic_weight_filter] = new_xdot_full
        self.p_xdot = pd.DataFrame(
            xdot_full, self.free_variable_names, ["data"], dtype=float
        )
        self.p_b["xdot"] = self.p_xdot
        self.p_b = self.p_b[["lb", "xdot", "ub"]]
        self.p_pure_xdot = deepcopy(self.p_xdot)
        self.p_pure_xdot[-num_constr:] = 0
        self.p_xdot_slack = deepcopy(self.p_xdot)
        self.p_xdot_slack[:-num_constr] = 0
        self.p_xdot_slack = self.p_xdot_slack.replace(np.nan, 0)
        if len(self.p_A) > 0:
            self.p_Ax = pd.DataFrame(
                self.p_A.dot(self.p_pure_xdot),
                self.inequality_constr_names,
                ["data"],
                dtype=float,
            )
            bA_slack = self.p_A.dot(self.p_xdot_slack)
            self.p_bA_raw.insert(1, "Ax", self.p_Ax)
            self.p_bA_raw.insert(2, "slack", bA_slack)
        else:
            self.p_Ax = pd.DataFrame()
        if len(self.p_E) > 0:
            self.p_Ex = pd.DataFrame(
                self.p_E.dot(self.p_pure_xdot),
                self.equality_constr_names,
                ["data"],
                dtype=float,
            )
            bA_slack = self.p_E.dot(self.p_xdot_slack)
            self.p_bE_raw.insert(1, "Ex w/o slack", self.p_Ex)
            self.p_bE_raw.insert(2, "slack", bA_slack)
        else:
            self.p_Ex = pd.DataFrame()

    @property
    def free_variable_names(self) -> list[str]:
        return self.degree_of_freedom_names + [
            c.name
            for c in self.qp_data_symbolic.constraint_collection.equality_constraints
        ]

    @property
    def degree_of_freedom_names(self) -> list[str]:
        names = []
        for derivative in ["vel", "jerk"]:
            for k in range(self.qp_data_symbolic.config.prediction_horizon):
                if (
                    derivative == "vel"
                    and k > self.qp_data_symbolic.config.prediction_horizon - 3
                ):
                    continue
                for dof in self.qp_data_symbolic.degrees_of_freedom:
                    names.append(f"{dof.name}_{derivative}_k_{k}")
        return names

    @property
    def equality_constr_names(self):
        return self.qp_data_symbolic.eq_constraint_names

    @property
    def inequality_constr_names(self):
        return self.qp_controller.qp_data_factory.qp_data._inequality_bounds.names

    def _print_iis(self):
        import pandas as pd

        def print_iis_matrix(
            row_filter: np.ndarray,
            column_filter: np.ndarray,
            matrix: pd.DataFrame,
            bounds: pd.DataFrame,
        ):
            if len(row_filter) == 0:
                return
            filtered_matrix = matrix.loc[row_filter, column_filter]
            filtered_matrix["bounds"] = bounds.loc[row_filter]
            print(filtered_matrix)

        result = self.qp_controller.qp_solver.analyze_infeasibility()
        if result is None:
            logger.info(
                f"Can only compute possible causes with gurobi, "
                f"but current solver is {self.qp_controller.config.qp_solver_id.name}."
            )
            return
        lb_ids, ub_ids, eq_ids, lbA_ids, ubA_ids = result
        b_ids = lb_ids | ub_ids
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", None
        ):
            logger.info("Irreducible Infeasible Subsystem:")
            logger.info("  Free variable bounds")
            free_variables = self.p_lb[b_ids]
            free_variables["ub"] = self.p_ub[b_ids]
            free_variables = free_variables.rename(columns={"data": "lb"})
            print(free_variables)
            logger.info("  Equality constraints:")
            print_iis_matrix(eq_ids, b_ids, self.p_E, self.p_bE)
            logger.info("  Inequality constraint lower bounds:")
            print_iis_matrix(lbA_ids, b_ids, self.p_A, self.p_lbA)
            logger.info("  Inequality constraint upper bounds:")
            print_iis_matrix(ubA_ids, b_ids, self.p_A, self.p_ubA)
