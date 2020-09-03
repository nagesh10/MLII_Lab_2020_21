import React ,{Component} from "react";
import {Link} from "react-router-dom";


class Assignment_component extends Component{
    constructor(props) {
        super(props);



    }


    render() {
        return (
            <div style={{padding:30}}>
            <div className="card" style={{width: 180}}>
                <div className="card-body">
                    <h5 className="card-title">{this.props.title}</h5>

                    <p className="card-text">{this.props.details}</p>
                    <Link to={this.props.link}>Open</Link>


                </div>
            </div>
                </div>

        );
    }

}
export default Assignment_component;

const styles = {
    assignment_base:{

        width:120,
        height:120,
        border:2,
        borderColor:'#99b6f2',
        backgroundColor:'#79b6f2',
        padding:30,




    },
    textstyle:{
        flex:1,
        textAlign:'center',

        justifyContent:'flex-end',


    }
}
