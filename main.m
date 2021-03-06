function main
    close all;
    clear;
    clc;

    inicio = cputime;

    global mejor_indiv;
    global mejor_valor;
    global caracteristicas;
    global class;
    global id;
    global VarName1;
    global VarName2;
    global VarName3;
    global VarName4;
    global VarName5;
    global VarName6;
    global VarName7;
    global VarName8;
    global VarName9;
    global VarName10;
    global VarName11;
    global VarName12;
    global VarName13;
    global VarName14;
    global VarName15;
    global VarName16;
    global VarName17;
    global VarName18;
    global VarName19;
    global VarName20;
    global VarName21;
    global VarName22;
    global VarName23;
    global VarName24;
    global VarName25;
    global VarName26;
    global VarName27;
    global VarName28;
    global VarName29;
    global VarName30;
    global VarName31;
    global VarName32;
    global VarName33;
    global VarName34;
    %Cantidad de lineas del data set wisconsin
    global linWin;
    %Cantidad de lineas del dataset indians
    global linInd;
    %Cantidad de lineas del dataset ionosphere
    global linIon;
    %Cantidad de lineas del data set german
    global linGer;
    %Cantidad de lineas del data set australian
    global linAus;

    linWin = 569;
    linInd = 768;
    linIon = 351;
    linGer = 1000;
    linAus = 690;
    
    mejor_indiv = [];
    mejor_valor = 0;
    
    %wisconsin
    %[id,class,VarName1,VarName2,VarName3,VarName4,VarName5,VarName6,VarName7,VarName8,VarName9,VarName10,VarName11,VarName12,VarName13,VarName14,VarName15,VarName16,VarName17,VarName18,VarName19,VarName20,VarName21,VarName22,VarName23,VarName24,VarName25,VarName26,VarName27,VarName28,VarName29, VarName30] = wdbc('wdbc.data.txt',1, 569);

    %pima indians
    [VarName1,VarName2,VarName3,VarName4,VarName5,VarName6,VarName7,VarName8,class] = pimaindians('pima-indians-diabetes.data.txt', 1, 768);

    %ionosphere
    %[VarName1,VarName2,VarName3,VarName4,VarName5,VarName6,VarName7,VarName8,VarName9,VarName10,VarName11,VarName12,VarName13,VarName14,VarName15,VarName16,VarName17,VarName18,VarName19,VarName20,VarName21,VarName22,VarName23,VarName24,VarName25,VarName26,VarName27,VarName28,VarName29,VarName30,VarName31,VarName32,VarName33,VarName34,class] = ionosphere('ionosphere.data.txt', 1, 351)
    %clc

    %german
    %[VarName1,VarName2,VarName3,VarName4,VarName5,VarName6,VarName7,VarName8,VarName9,VarName10,VarName11,VarName12,VarName13,VarName14,VarName15,VarName16,VarName17,VarName18,VarName19,VarName20,VarName21,VarName22,VarName23,VarName24,class] = german('german.data-numeric.txt', 1, 1000);

    %australian
    %[VarName1,VarName2,VarName3,VarName4,VarName5,VarName6,VarName7,VarName8,VarName9,VarName10,VarName11,VarName12,VarName13,VarName14,class] = australian('australian.dat.txt', 1, 690);

    %% Caracteristicas
    %wisconsin
    %caracteristicas = 30; 
    %pima indians
    caracteristicas = 8; 
    %ionosphere
    %caracteristicas = 34; 
    %german
    %caracteristicas = 24; 
    %australian
    %caracteristicas = 14; 
    %% SVM
    % 2 parametros C y sigma
    parametros = 2;
    %% Parametros de poblacion
    %Tama�o de poblacion inicial
    %pobInicial=input('Ingrese tama�o inicial de poblacion(50 default): ');    
    options = gaoptimset;
    %Parametros de Poblacion
    options = gaoptimset(options,'PopulationSize' , 50);
    %Poblacion Inicial Random
    poblacion = (rand(options.PopulationSize, caracteristicas+(8*parametros))>0.5);
    %Tipo de Poblacion
    options = gaoptimset(options, 'PopulationType' , 'bitstring');
    options = gaoptimset(options, 'InitialPopulation', poblacion); 
    %% Criterios de parada 
    %Numero de generaciones
    %gen=input('Ingrese cantidad generaciones: ');
    options = gaoptimset(options, 'Generations', 100);
    options = gaoptimset(options,'StallGenLimit', 100);
    %% Seleccion por ruleta
    options = gaoptimset(options, 'SelectionFcn', {@selectionroulette});
    %% Cruzamiento
    % Cruzamiento en 2 puntos
    options = gaoptimset(options, 'CrossoverFcn', {@crossovertwopoint});
    % Fraccion de cruzamiento
    %crossFract=input('Fraccion de la siguiente generacion generada por crossover (0.8 default): ');
    options = gaoptimset(options, 'CrossoverFraction', 0.9); 
    %% Mutacion
    % Mutacion Uniforme
    %mutRate=input('Probabilidad de mutacion (Uniforme default 0.01): ');
    options = gaoptimset(options ,'MutationFcn', {@mutationuniform, 0.09}); 
    %% Grafico
    options = gaoptimset(options, 'Display', 'off'); 
    options = gaoptimset(options, 'PlotInterval', 1); 
    options = gaoptimset(options,'PlotFcns',{@gaplotbestf}); 
    %% GA
    [individuo, Fval, ~, Output, population, ~]= ga(@Fitness,caracteristicas+(parametros*8),options);

    total = cputime - inicio;
    fprintf('Tiempo computacional : %g\n', total);
    fprintf('The best gen was: ');
    disp(individuo);
    fprintf('The best fitness was: %d\n', mejor_valor);
    fprintf('The number of generations was : %d\n', Output.generations);
    fprintf('The number of function evaluations was : %d\n', Output.funccount);
    fprintf('The best function value found was : %g\n', Fval);
end
%% Funcion Fitness
function valor = Fitness(string) 
    global mejor_valor;
    global mejor_indiv;
    global caracteristicas;
    global class;
    global id;
    global VarName1;
    global VarName2;
    global VarName3;
    global VarName4;
    global VarName5;
    global VarName6;
    global VarName7;
    global VarName8;
    global VarName9;
    global VarName10;
    global VarName11;
    global VarName12;
    global VarName13;
    global VarName14;
    global VarName15;
    global VarName16;
    global VarName17;
    global VarName18;
    global VarName19;
    global VarName20;
    global VarName21;
    global VarName22;
    global VarName23;
    global VarName24;
    global VarName25;
    global VarName26;
    global VarName27;
    global VarName28;
    global VarName29;
    global VarName30;
    global VarName31;
    global VarName32;
    global VarName33;
    global VarName34;
    global linWin;
    global linInd;
    global linIon;
    global linGer;
    global linAus;
    %Si las caracteristicas son todas 0, no se calcula fitness
    if any(string(1:caracteristicas))
        %wisconsin
        %dataset = [VarName1,VarName2,VarName3,VarName4,VarName5,VarName6,VarName7,VarName8,VarName9,VarName10,VarName11,VarName12,VarName13,VarName14,VarName15,VarName16,VarName17,VarName18,VarName19,VarName20,VarName21,VarName22,VarName23,VarName24,VarName25,VarName26,VarName27,VarName28,VarName29, VarName30];
        %pima indians
        dataset = [VarName1,VarName2,VarName3,VarName4,VarName5,VarName6,VarName7,VarName8];
        %ionosphere
        %dataset = [VarName1,VarName2,VarName3,VarName4,VarName5,VarName6,VarName7,VarName8,VarName9,VarName10,VarName11,VarName12,VarName13,VarName14,VarName15,VarName16,VarName17,VarName18,VarName19,VarName20,VarName21,VarName22,VarName23,VarName24,VarName25,VarName26,VarName27,VarName28,VarName29,VarName30,VarName31,VarName32,VarName33,VarName34];
        %clc
        %german
        %dataset = [VarName1,VarName2,VarName3,VarName4,VarName5,VarName6,VarName7,VarName8,VarName9,VarName10,VarName11,VarName12,VarName13,VarName14,VarName15,VarName16,VarName17,VarName18,VarName19,VarName20,VarName21,VarName22,VarName23,VarName24];
        %australian
        %dataset = [VarName1,VarName2,VarName3,VarName4,VarName5,VarName6,VarName7,VarName8,VarName9,VarName10,VarName11,VarName12,VarName13,VarName14];
        
        nuevoDataset = [];
        for i=1:caracteristicas;
            fila = [];
            for j=(((i-1)*linInd)+1):linInd*i;
                fila = [fila dataset(j)];
            end
            if(string(i) == 1);
                %Columna de caracteristicas
                nuevoDataset = [nuevoDataset fila'];
            end
        end
        %String de parametros C (8 bits siguientes a la ultima caracteristica)
        stringC = string(caracteristicas+1:caracteristicas+8);
        c = 0;
        aux = 0;
        for i=8:-1:1;
            c = c + (stringC(i)*(2^(aux)));
            aux = aux + 1;
        end

        if (c == 0);
            c = 1;
        end
        
        %String de parametros de Sigma
        stringSigma = string(caracteristicas+9:caracteristicas+16);
        sigma = 0;
        aux = 0;
        for i=8:-1:1;
            sigma = sigma + (stringSigma(i)*(2^(aux)));
            aux = aux + 1;
        end
        
        if (sigma == 0);
            sigma = 1;
        end
        
        valor = svm_classifier(nuevoDataset,class,sigma,c);
        valor = -valor;
        
        if mejor_valor > valor
            mejor_indiv = string;
            mejor_valor = valor;
        end
    
    else
        valor = 0;
    end
end