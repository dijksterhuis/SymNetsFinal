is.odd <- function(x) {return(x %% 2 != 0) }

col_desc <- function(inputdata)
{
        col_names = colnames(inputdata)
        n_cols = length(col_names)
        return(list(n_cols=n_cols,col_names=col_names))
}

get_data <- function(inputdata,h_act_filter,y_act_filter)
{
data.df = data.frame(read.table(inputdata, sep = "|")[2])
n_cols = col_desc(read.table(inputdata, sep = "|"))$n_cols
#print(n_cols)
for (i in 2:n_cols)
        {
                if( i %% 2 == 0 && i < 22)
                {
                        v <- read.table(inputdata, sep = "|")[i]
                        col_naming <- paste(read.table(inputdata, sep = "|")[i-1][1,1])
                        data.df[,col_naming] <- v
                }
                else if (i >= 22)
                {
                        v <- read.table(inputdata, sep = "|")[i]
                        col_name <- paste("results",i-21,sep = "_")
                        data.df[,col_name] <- v
                }
}
        data.df <- subset(data.df,select = -V2)
        data.df$mean_accs <- rowMeans(data.df[,11:13])
        return(data.df)
}


plot_2_continuous_params <- function(input_file,in1="rate",in2="width",in3="mean_accs",contour_bins = 12,filter_type="none",filter1="sigmoid",filter2 = "sigmoid",save=FALSE)
{
        library(ggplot2) ; library(reshape) 
        
        data <- get_data(input_file)
        names <- col_desc(data)$col_names

        if (filter_type == 'acts'){
                data <- subset(data, hidden_act == filter1)
                data <- subset(data, y_act == filter2)
                plotname = paste(strsplit(strsplit(input_file,"/")[[1]][5],"_")[[1]][1],filter1,filter2,sep=" ")
                }
        else if (filter_type == 'opts'){
                data <- subset(data,opt == filter1)
                plotname = paste(strsplit(strsplit(input_file,"/")[[1]][5],"_")[[1]][1], filter1,sep=" ")
        }
        else if (filter_type == 'loss'){
                data <- subset(data,loss == filter1)
                plotname = paste(strsplit(strsplit(input_file,"/")[[1]][5],"_")[[1]][1], filter1,sep=" ")
        }
        else{plotname = paste(strsplit(strsplit(input_file,"/")[[1]][5],"_")[[1]][1],sep=" ")}
        print('Dimensions')
        print(dim(data))
        
        tmp.df = data.frame(var_1 = data[[in1]], var_2 = data[[in2]], var_3 = data[[in3]])
        
        sizes_ceil = signif((2*log10(max(tmp.df$var_2)))*(2*log10(max(tmp.df$var_2))),1)
        sizes_ceil = 20
        sizes_floor = floor((2*log10(max(tmp.df$var_2)))*(2*log10(max(tmp.df$var_2))))
        p <- ggplot(tmp.df , aes(x = var_1, y = var_2, z = var_3 )) + 
                geom_raster(aes(fill = var_3),interpolate = TRUE) +
                #geom_tile(aes(colour = var_3),size = sizes_ceil) +
                geom_contour(aes(z=var_3),color = "white",bins = contour_bins,alpha = 0.5) +
                geom_text(aes(label = floor(var_3)),colour = "white",alpha = 1) +
                xlab(in1) + ylab(in2) + ggtitle(paste(plotname)) + 
                #scale_size_continuous(limits = c(50,100), name = in3) +
                scale_fill_continuous(limits = c(45,100), name = in3,low="dark blue",high="light blue") +
                scale_colour_continuous(limits = c(45,100), name = in3,low="dark blue",high="light blue") +
                scale_x_continuous(breaks = seq(round(min(tmp.df$var_1),1),round(max(tmp.df$var_1),1),1)) +
                scale_y_continuous(breaks = seq(0,max(tmp.df$var_2),sizes_floor)) 
        
        
        print(p)
        
        if (save == TRUE){ggsave(filename=paste(plotname,".jpg",sep=""), plot=p)}
}


acts_plots <- function(file,h_acts,y_acts,save=save){
        
        for (i in 1:length(y_acts)){
                for (j in 1:length(h_acts)){
                        plot_2_continuous_params(file,filter_type = 'acts',filter1=h_acts[j],filter2=y_acts[i],save=save)
                }
        }
        
}

optimiser_plots <- function(file,opts=c('ADAgrad','ADAdelta','PAD'),save=save){
        for (i in 1:length(opts)){
                plot_2_continuous_params(file,filter_type = 'opts',filter1=opts[i],save=save)
        }
}

loss_plots <- function(file,loss=c('hinge','mse','log'),save=FALSE){
        for (i in 1:length(loss)){
                plot_2_continuous_params(file,filter_type = 'loss',filter1=loss[i],save=save)
        }
}


datasetsize_plot <- function(file){
        library(reshape)
        library(ggplot2)
        
        data.df <- data.frame(read.csv(file,header = TRUE))
        
        tmp.df <- melt(data.df, id.vars=c("Epoch"),variable_name="Datasize")
        
        p <- ggplot(tmp.df, aes(x = Epoch , y = value, labels = Datasize, color = Datasize)) + geom_line()
        print(p)
}


activation_demos <- function(){
        
        source("http://peterhaschke.com/Code/multiplot.R")
        library(reshape)
        library(ggplot2)
        
        all_plots.list = list()
        
        x = seq(-10,10,0.1)
        sig_y = 1 / (1 + exp(-x))
        tanh_y = tanh(x)
        relu_xmod = x
        relu_xmod[relu_xmod<0] = 0
        relu_y = relu_xmod
        elu_y = c(exp(x[x<0]) - 1 ,x[x>=0])
        softplus = log(exp(x) + 1)
        softsign = x / (abs(x) + 1)
        softmax = exp(x) / sum(exp(x))
        relu6 = c(relu_xmod[relu_xmod<6] , rep(0, length(seq(6,10,0.1))))
        
        graf.df <- data.frame("x" = x, "sigmoid" = sig_y, "tanh"= tanh_y,"softsign"=softsign, "softmax" = softmax, "relu" = relu_y,"relu6" = relu6, "elu"=elu_y, "softplus" = softplus ,check.rows = FALSE)
        props = col_desc(graf.df)
        
        for ( i in 2:props$n_cols[1]){
                tmp.df = data.frame( X = graf.df$x , Y = graf.df[,i] )
                p = ggplot(tmp.df, aes(x = X , y = Y))
                p <- p + geom_line(colour="blue") + ggtitle(paste(props$col_names[i])) + xlim(-10,10)
                if (max(tmp.df$Y) > 1){
                        p <- p + ylim(-10, 10)
                }
                else if(max(tmp.df$Y)>0.5){
                        p <- p + ylim(-1, 1)
                }
                else
                {
                maxmin = round(max(tmp.df$Y),2)
                p <- p + ylim(-maxmin, maxmin)
                }
                all_plots.list[i-1] <- list(p)
        }
        multiplot(plotlist=all_plots.list,cols = 2)
}

plot_2_noncontinuous_params <- function(input_file,in1="rate",in2="width",in3="mean_accs",contour_bins = 12,filter_type="none",filter1="sigmoid",filter2 = "sigmoid",save=FALSE)
{
        library(ggplot2) ; library(reshape) 
        
        data <- get_data(input_file)
        names <- col_desc(data)$col_names
        
        if (filter_type == 'acts'){
                data <- subset(data, hidden_act == filter1)
                data <- subset(data, y_act == filter2)
                plotname = paste(strsplit(strsplit(input_file,"/")[[1]][5],"_")[[1]][1],filter1,filter2,sep=" ")
        }
        else if (filter_type == 'opts'){
                data <- subset(data,opt == filter1)
                plotname = paste(strsplit(strsplit(input_file,"/")[[1]][5],"_")[[1]][1], filter1,sep=" ")
        }
        else if (filter_type == 'loss'){
                data <- subset(data,loss == filter1)
                plotname = paste(strsplit(strsplit(input_file,"/")[[1]][5],"_")[[1]][1], filter1,sep=" ")
        }
        else{plotname = paste(strsplit(strsplit(input_file,"/")[[1]][4],"_")[[1]][1],sep=" ")}
        print('Dimensions')
        print(dim(data))
        
        tmp.df = data.frame(var_1 = data[[in1]], var_2 = data[[in2]], var_3 = data[[in3]])
        
        sizes_ceil = signif((2*log10(max(tmp.df$var_2)))*(2*log10(max(tmp.df$var_2))),1)
        sizes_floor = floor((2*log10(max(tmp.df$var_2)))*(2*log10(max(tmp.df$var_2))))
        p <- ggplot(tmp.df , aes(x = var_1, y = var_2, z = var_3 )) + 
                geom_point(aes(color = var_3),size = sizes_ceil) +
                #geom_tile(aes(colour = var_3),size = sizes_ceil) +
                #geom_contour(aes(z=var_3),color = "white",bins = contour_bins,alpha = 0.5) +
                geom_text(aes(label = floor(var_3)),colour = "white",alpha = 1) +
                xlab(in1) + ylab(in2) + ggtitle(paste(plotname)) + 
                scale_size_continuous(limits = c(45,100), name = in3) +
                scale_colour_continuous(limits = c(45,100), name = in3,low="dark blue",high="light blue") +
                #scale_x_continuous(breaks = seq(0,1,0.1)) +
                scale_y_continuous(breaks = seq(0,max(tmp.df$var_2),5)) 
        
        
        print(p)
        
        if (save == TRUE){ggsave(filename=paste(plotname,".jpg",sep=""), plot=p)}
}