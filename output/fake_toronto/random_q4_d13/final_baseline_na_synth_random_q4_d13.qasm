OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.7120078) q[23];
rz(4.901253) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(0.54093528) q[25];
rz(3.0274815) q[25];
rz(1.2017439) q[25];
rz(6.1477657) q[25];
rz(2.2261368) q[25];
rz(3.5231617) q[25];
rz(1.5094849) q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[25],q[24];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[24],q[25];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[25],q[26];
cx q[25],q[24];
rz(3.4182651) q[24];
rz(2.9895602) q[25];
rz(0.7042347) q[26];
rz(0.71350104) q[26];
rz(0.54075859) q[26];
rz(2.2158505) q[26];
rz(4.6562528) q[26];
rz(1.0033839) q[26];
rz(1.0754204) q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[24],q[25];
cx q[24],q[23];
cx q[23],q[24];
rz(1.4459097) q[23];
rz(3.107316) q[24];
rz(4.5529028) q[24];
rz(5.0352815) q[24];
rz(3.6715963) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[25],q[24];
rz(2.5708871) q[24];
rz(6.242467) q[25];
rz(2.0586653) q[25];
