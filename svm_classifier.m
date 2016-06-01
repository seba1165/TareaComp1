%funcion que retorna al precisi?n alcanzada por la SVM
function fit = svm_classifier(data,class,sigma,c)

%wisconsin
%species = cellstr(class);
%groups = ismember(species,'M');

%ionosphere
species = cellstr(class);
groups = ismember(species,'g');

%% Randomly select training and test sets.
%indians, german, australian 
%[train, test] = crossvalind('holdOut',class);
%cp = classperf(class);
%ionosphere, wisconsin 
[train, test] = crossvalind('holdOut',groups);
cp = classperf(groups);

%% Use the svmtrain function to train an SVM classifier using a radial basis function and plot the grouped data.
%svmStruct = svmtrain(data(train,:),groups(train),'showplot',true);
%svmStruct = svmtrain(data(train,:),class(train),'showplot',true,'kernel_function','rbf','rbf_sigma',sigma,'AUTOSCALE',true,'BoxConstraint',c);

options = optimset('maxiter',10000,'largescale','off','TolX',5e-4,'TolFun',5e-4);

%svmStruct = svmtrain(data(train,:),class(train),'showplot',false,'kernel_function','rbf','AUTOSCALE',true,'rbf_sigma',sigma,'BoxConstraint',c,'Method','QP','quadprog_opts',options);
svmStruct = svmtrain(data(train,:),groups(train),'showplot',false,'kernel_function','rbf','AUTOSCALE',true,'rbf_sigma',sigma,'BoxConstraint',c,'Method','QP','quadprog_opts',options);
%svmStruct = svmtrain(data(train,:),groups(train),'showplot',true,'kernel_function','polynomial');

%Se clasifica el conjunto de test en base al modelo obtenido
classes = svmclassify(svmStruct,data(test,:),'showplot',false);

%se evalua el performance del modelo
classperf(cp,classes,test);
fit = cp.CorrectRate;
%fit = get(cp)

end