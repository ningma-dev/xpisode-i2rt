//! Recursive Newton-Euler for a serial chain of revolute joints. No allocation.

use crate::config::Link;
use crate::MAX_MOTORS;

type V = [f64; 3];
type M = [[f64; 3]; 3];

fn cross(a: V, b: V) -> V {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V, s: f64) -> V {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn mul(m: &M, v: V) -> V {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}
fn mul_transpose(m: &M, v: V) -> V {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}
fn matmul(a: &M, b: &M) -> M {
    let mut c = [[0.0; 3]; 3];
    for (i, row) in c.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}
fn from_quaternion(q: [f64; 4]) -> M {
    let [w, x, y, z] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}
fn from_axis_angle(a: V, angle: f64) -> M {
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [t * a[0] * a[0] + c, t * a[0] * a[1] - s * a[2], t * a[0] * a[2] + s * a[1]],
        [t * a[0] * a[1] + s * a[2], t * a[1] * a[1] + c, t * a[1] * a[2] - s * a[0]],
        [t * a[0] * a[2] - s * a[1], t * a[1] * a[2] + s * a[0], t * a[2] * a[2] + c],
    ]
}

#[derive(Clone, Debug)]
struct Body {
    position: V,
    rest: M,
    axis: V,
    mass: f64,
    com: V,
    inertia: M,
}

#[derive(Clone, Debug)]
pub struct Chain {
    bodies: Vec<Body>,
    gravity: V,
}

impl Chain {
    pub fn new(links: &[Link], gravity: [f64; 3]) -> Chain {
        assert!(links.len() <= MAX_MOTORS);
        let bodies = links
            .iter()
            .map(|l| {
                let [xx, yy, zz, xy, xz, yz] = l.inertia;
                Body {
                    position: l.position,
                    rest: from_quaternion(l.quaternion),
                    axis: l.axis,
                    mass: l.mass,
                    com: l.center_of_mass,
                    inertia: [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
                }
            })
            .collect();
        Chain { bodies, gravity }
    }

    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// The joint torques for these positions, velocities and accelerations, with or without gravity.
    pub fn inverse_dynamics(
        &self,
        q: &[f64],
        qd: &[f64],
        qdd: &[f64],
        with_gravity: bool,
        tau: &mut [f64],
    ) {
        let n = self.bodies.len();
        let mut rotation = [[[0.0; 3]; 3]; MAX_MOTORS];
        let mut force = [[0.0; 3]; MAX_MOTORS];
        let mut moment = [[0.0; 3]; MAX_MOTORS];
        let mut w = [0.0; 3];
        let mut dw = [0.0; 3];
        let mut a = if with_gravity { scale(self.gravity, -1.0) } else { [0.0; 3] };
        for i in 0..n {
            let b = &self.bodies[i];
            let r = matmul(&b.rest, &from_axis_angle(b.axis, q[i]));
            rotation[i] = r;
            let a_at_joint = add(a, add(cross(dw, b.position), cross(w, cross(w, b.position))));
            let w_parent = mul_transpose(&r, w);
            w = add(w_parent, scale(b.axis, qd[i]));
            dw = add(
                add(mul_transpose(&r, dw), scale(b.axis, qdd[i])),
                cross(w_parent, scale(b.axis, qd[i])),
            );
            a = mul_transpose(&r, a_at_joint);
            let a_com = add(a, add(cross(dw, b.com), cross(w, cross(w, b.com))));
            force[i] = scale(a_com, b.mass);
            moment[i] = add(mul(&b.inertia, dw), cross(w, mul(&b.inertia, w)));
        }
        let mut f_child = [0.0; 3];
        let mut n_child = [0.0; 3];
        for i in (0..n).rev() {
            let b = &self.bodies[i];
            let mut f = force[i];
            let mut m = add(moment[i], cross(b.com, force[i]));
            if i + 1 < n {
                let child = &self.bodies[i + 1];
                let f_here = mul(&rotation[i + 1], f_child);
                f = add(f, f_here);
                m = add(m, add(mul(&rotation[i + 1], n_child), cross(child.position, f_here)));
            }
            tau[i] = dot(b.axis, m);
            f_child = f;
            n_child = m;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point_mass_link(axis: V, position: V, com: V, mass: f64) -> Link {
        Link {
            position,
            quaternion: [1.0, 0.0, 0.0, 0.0],
            axis,
            mass,
            center_of_mass: com,
            inertia: [0.0; 6],
        }
    }

    #[test]
    fn a_pendulum_has_the_torque_it_should() {
        // One link, mass 2 at 0.5 along x, turning about y, gravity along -z.
        let link = point_mass_link([0.0, 1.0, 0.0], [0.0; 3], [0.5, 0.0, 0.0], 2.0);
        let chain = Chain::new(&[link], [0.0, 0.0, -9.81]);
        let mut tau = [0.0];
        // Horizontal: gravity pulls about +y with m g l.
        chain.inverse_dynamics(&[0.0], &[0.0], &[0.0], true, &mut tau);
        assert!((tau[0] + 2.0 * 9.81 * 0.5).abs() < 1e-9, "{}", tau[0]);
        // Hanging straight down at q = pi / 2 about +y (x goes to -z): no torque.
        chain.inverse_dynamics(&[std::f64::consts::FRAC_PI_2], &[0.0], &[0.0], true, &mut tau);
        assert!(tau[0].abs() < 1e-9, "{}", tau[0]);
        // Without gravity, acceleration needs m l^2 qdd.
        chain.inverse_dynamics(&[0.3], &[1.0], &[3.0], false, &mut tau);
        assert!((tau[0] - 2.0 * 0.25 * 3.0).abs() < 1e-9, "{}", tau[0]);
    }
}
