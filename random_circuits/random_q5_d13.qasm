OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(0.75501486) q[4];
cx q[1],q[0];
rz(2.0705272) q[3];
rz(3.9791243) q[2];
rz(5.7354891) q[2];
cx q[4],q[1];
cx q[0],q[3];
cx q[4],q[1];
cx q[0],q[3];
rz(5.1755934) q[2];
cx q[2],q[1];
cx q[3],q[4];
rz(1.7225174) q[0];
cx q[4],q[0];
rz(3.8708919) q[1];
rz(4.2089078) q[2];
rz(1.3195483) q[3];
rz(6.2332295) q[3];
rz(4.6242411) q[0];
rz(3.4144046) q[1];
rz(2.3180908) q[2];
rz(3.846423) q[4];
cx q[3],q[2];
rz(6.0086954) q[0];
cx q[1],q[4];
cx q[3],q[4];
rz(5.1124681) q[0];
rz(4.8663147) q[2];
rz(5.8191892) q[1];
cx q[3],q[2];
rz(3.5476596) q[1];
rz(5.369761) q[0];
rz(5.6246136) q[4];
cx q[4],q[1];
rz(4.9141292) q[2];
rz(3.559654) q[0];
rz(5.1121798) q[3];
cx q[0],q[3];
rz(3.7402514) q[1];
rz(4.163634) q[2];
rz(1.7307449) q[4];
rz(0.111389) q[4];
cx q[0],q[1];
cx q[2],q[3];
cx q[1],q[0];
rz(2.7662384) q[2];
rz(2.1478644) q[3];
rz(5.6768025) q[4];
