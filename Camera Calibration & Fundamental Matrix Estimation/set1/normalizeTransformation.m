function[T] = normalizeTransformation(q)

t = diag([1 1 1]);
t(1,3) = -(mean(q(:,1)));
t(2,3) = - (mean(q(:,2))); 

s = diag([1 1 1]);
s(1,1) = 1/(std(q(:,1)));
s(2,2) = 1/(std(q(:,2)));

T = s*t;
end
