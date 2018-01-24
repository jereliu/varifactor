u <- seq(-10, 10, 0.01)
v <- seq(-10, 10, 0.01)

post_surface <- 
  function(u, v, y = 2500, 
           lambda = 1, eps = 1, 
           family= "gaus", n = 100,
           plot_type = "persp",
           ...)
  {
    # prior
    log_prior <- outer(u^2, v^2, "+")
    
    # likelihood
    theta <- 
      u %*% t(v) + 
      eps * matrix(rnorm(length(u) * length(v)), 
                   nrow = length(u), ncol = length(v))
    if (family == "gaus") {
      A = theta^2/2
    } else if (family == "pois"){
      A = exp(theta)
    } else if (family == "bern"){
      A = n * log(1 + exp(theta))
    } else if (family == "null"){
      A = 0
    }
    
    log_likhd <- - y * theta + A
    
    # posterior
    log_post <- log_likhd + lambda * log_prior
    
    #image(log_post, col = heat.colors(20))
    if (plot_type == "persp"){
      persp(u, v, log_post, ticktype = "detailed", ...)
    } else if (plot_type == "image"){
      image(u, v, log_post, ...)
      min_id <- which(log_post == min(log_post), arr.ind = TRUE)
      points(matrix(u[min_id], ncol = 2), pch = 19, cex = 0.5)
    }
    
    log_post
  }

log_post <- 
  post_surface(u, v, y = 10, lambda = 0, eps = 0, 
               family= "gaus", 
               plot_type = "image", zlim = c(-10000, 10000), 
               main = "gaus")
               #phi = 30, theta = 15, zlim = c(-1000, 1000), main = "gaus")

log_post <- 
  post_surface(u, v, y = 10, lambda = 0, eps = 0, 
               family= "bern", n = 100,
               plot_type = "image", zlim = c(-10000, 10000), 
               main = "bern")

log_post <- 
  post_surface(u, v, y = 10, lambda = 0, eps = 0, 
               family= "pois", 
               plot_type = "image", zlim = c(-10000, 10000), 
               main = "pois")
